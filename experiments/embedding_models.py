"""Which image-text embedding model should the robot's visual memory use? Measured on this machine.

The visual memory stores an embedding of every object crop the robot sees and later finds them by text
("bottle", "something to drink from"). That needs (1) retrieval that holds on small, low-angle crops, rendered
and real, (2) one fixed rule that says "this crop fits this phrase" across phrases, and (3) little memory and
latency next to the LLM, Whisper and YOLO. This harness measures all three for a few candidates, each in a
fresh process so that memory figures don't mix. Results and docs: docs/visual-memory-models.md.

    .venv/bin/python experiments/embedding_models.py fetch            # pinned weights -> _deps/embedding-cache
    .venv/bin/python experiments/embedding_models.py collect          # sim + COCO crops (labelled)
    .venv/bin/python experiments/embedding_models.py embed            # every model: timing, memory, embeddings
    .venv/bin/python experiments/embedding_models.py report           # retrieval, decision rules, tables
    .venv/bin/python experiments/embedding_models.py verify [--device cpu]   # the shipped go2/embedder.py

`embed siglip2-b16@192` runs a model at another input size (position embeddings interpolated). Everything
downloaded or produced lives under _deps/embedding-cache/ (git-ignored). MobileCLIP2 needs open_clip and timm,
which the project does not install: it runs from a throwaway venv (_deps/embedding-venv, made with
_deps/tools/uv; see the docs), and is skipped when that is missing.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "_deps" / "embedding-cache"
CROPS = CACHE / "crops"
RESULTS = CACHE / "results"
SIDE_VENV = ROOT / "_deps" / "embedding-venv" / "bin" / "python"

# name -> how to get and run it. Revisions are exact Hugging Face commits; the OpenAI URL carries its sha256.
CANDIDATES = {
    "siglip-b16": dict(kind="siglip", repo="google/siglip-base-patch16-224",
                       revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", licence="Apache-2.0"),
    "siglip2-b16": dict(kind="siglip", repo="google/siglip2-base-patch16-224",
                        revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", licence="Apache-2.0"),
    "clip-b16": dict(kind="openai", repo="ViT-B/16", licence="MIT (OpenAI CLIP repository)",
                     url="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6"
                         "937f9e85e4ecb61988df416f/ViT-B-16.pt"),
    "mobileclip2-b": dict(kind="openclip", repo="timm/MobileCLIP2-B-OpenCLIP", arch="MobileCLIP2-B",
                          revision="6a3bc3dc48d04504ecfca54f8b26bd78a71ddc4e",
                          licence="Apple ML Research Model License (research only)"),
}
HF_FILES = ["config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "open_clip_config.json", "open_clip_model.safetensors", "vocab.json",
            "merges.txt"]

# ---- the data's labels and the queries --------------------------------------------------------------------
# Simulator labels are the COCO names YOLO would give; the toys get what they are, for the queries.
SIM_LABELS = {"dog": "toy dog", "elephant": "toy elephant", "horse": "toy horse", "truck": "toy truck",
              "sports ball": "ball", "box": "cardboard box"}
SIM_SKIP = {"wall", "floor", "door"}
COCO_CLASSES = {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                "orange", "broccoli", "carrot", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "toothbrush", "backpack", "handbag", "suitcase", "sports ball", "cat", "dog", "person", "bench"}
COCO_LABELS = {"sports ball": "ball"}
FOOD = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake"}
TOYS = {"teddy bear", "toy dog", "toy elephant", "toy horse", "toy truck"}
DRINK = {"cup", "wine glass", "bottle"}
# (query, type, labels that fit, labels that are neither right nor wrong). Exact-label queries are added from
# the data. "none": nothing in the data fits, so every acceptance is a false alarm.
QUERIES = [
    ("sofa", "synonym", {"couch"}, set()),
    ("mug", "synonym", {"cup"}, {"wine glass"}),
    ("television", "synonym", {"tv"}, {"laptop"}),
    ("fridge", "synonym", {"refrigerator"}, set()),
    ("plant", "synonym", {"potted plant"}, {"broccoli", "vase"}),
    ("table", "synonym", {"dining table", "desk"}, set()),
    ("stuffed animal", "synonym", TOYS - {"toy truck"}, set()),
    ("bag", "synonym", {"backpack", "handbag", "suitcase"}, set()),
    ("something to drink from", "loose", DRINK, {"bowl", "vase"}),
    ("a toy", "loose", TOYS | {"ball"}, set()),
    ("somewhere to sit", "loose", {"chair", "couch", "bench"}, {"bed", "toilet", "dining table"}),
    ("food", "loose", FOOD, {"bowl", "dining table"}),
    ("a screen", "loose", {"tv", "laptop"}, {"cell phone", "microwave"}),
    ("something to read", "loose", {"book"}, set()),
    ("a pet", "loose", {"cat", "dog"}, {"toy dog", "teddy bear", "toy elephant", "toy horse"}),
    ("kitchen appliance", "loose", {"refrigerator", "oven", "microwave", "toaster"}, {"sink"}),
    ("bottle", "related", DRINK, set()),             # the motivating case: containers ahead of the rest
    ("a drink", "related", DRINK, {"bowl"}),
    ("rubbish bin", "none", set(), set()),
    ("my shoes", "none", set(), set()),
    ("a guitar", "none", set(), set()),
    ("car keys", "none", set(), set()),
    ("a fire extinguisher", "none", set(), set()),
    ("a vacuum cleaner", "none", set(), set()),
    ("a bicycle", "none", set(), set()),
]
EXACT_IGNORE = {"dining table": {"desk"}, "desk": {"dining table"}, "toy dog": {"dog"}, "dog": {"toy dog"},
                "cup": {"wine glass"}, "wine glass": {"cup"}}
BACKGROUND = ["an object", "something", "the floor", "a wall", "the inside of a room", "a blurry image"]
DETERMINERS = {"a", "an", "the", "my", "your", "our", "some", "something", "somewhere", "anything", "this", "that"}
MASS = {"food", "furniture", "rubbish", "clothing", "laundry", "water"}
TEMPLATES = {"plain": ["{}"], "photo": ["a photo of {}."], "siglip": ["this is a photo of {}."],
             "ensemble": ["a photo of {}.", "a photo of {} on the floor.", "a low-angle photo of {}.",
                          "a blurry photo of {}.", "a close-up photo of {}.", "a photo of {} in a home."]}
MODES = ("squash", "pad", "crop")         # how a crop becomes a square: stretch, pad with grey, centre-cut


def with_article(phrase):
    first = phrase.split()[0]
    if first in DETERMINERS or phrase in MASS:
        return phrase
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


def prompts(phrases, template):
    """[phrase] -> [[prompt]] for one template (several per phrase for an ensemble). Lowercased: SigLIP 2's
    tokenizer does not lowercase by itself and the models were trained on lowercased text."""
    return [[t.format(p.lower() if template == "plain" else with_article(p.lower())) for t in TEMPLATES[template]]
            for p in phrases]


# ---- memory --------------------------------------------------------------------------------------------------
def footprint_mb():
    """macOS phys_footprint (what Activity Monitor shows as Memory, including Metal/MPS buffers), in MB, or None.
    proc_pid_rusage(RUSAGE_INFO_V2): ri_phys_footprint is the 9th 64-bit field after a 16-byte uuid."""
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        buf = (ctypes.c_uint64 * 40)()
        if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) != 0:
            return None
        return buf[2 + 7] / 2**20
    except OSError:
        return None


def memory():
    import psutil
    return {"rss_mb": psutil.Process().memory_info().rss / 2**20, "footprint_mb": footprint_mb()}


# ---- preprocessing (ours, the same for every model, deterministic) ----------------------------------------
def square(crop, size, mode):
    """HxWx3 uint8 -> size x size x 3 float32 in [0, 1]."""
    from PIL import Image
    img = Image.fromarray(np.ascontiguousarray(crop))
    w, h = img.size
    if mode == "pad":
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas
    elif mode == "crop":
        side = min(w, h)
        img = img.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
    return np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255


# ---- the candidates, behind one small interface --------------------------------------------------------------
def snapshot(spec):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(spec["repo"], revision=spec["revision"], cache_dir=CACHE / "hf",
                                  allow_patterns=HF_FILES, local_files_only=True))


class Candidate:
    """encode_images(float32 NxSxSx3) and encode_texts([[prompt]]) -> unit vectors; scale/bias: the model's own
    logit parameters (CLIP-style models have no bias)."""

    def __init__(self, name, device, dtype):
        import torch
        self.name, self.spec, self.device = name, CANDIDATES[name.split("@")[0]], device
        self.dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype]
        getattr(self, "_load_" + self.spec["kind"])()
        self.model.eval()

    def _load_siglip(self):
        import torch
        from transformers import AutoModel, PreTrainedTokenizerFast
        path = snapshot(self.spec)
        self.model = AutoModel.from_pretrained(path, dtype=self.dtype).to(self.device)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(path)   # tokenizer.json: no sentencepiece
        # "siglip-b16@160": a smaller input than trained (position embeddings interpolated) for speed
        native = self.model.config.vision_config.image_size
        self.size = int(self.name.split("@")[1]) if "@" in self.name else native
        self.mean = self.std = np.array([.5, .5, .5], np.float32)
        self.scale = float(self.model.logit_scale.exp())
        self.bias = float(self.model.logit_bias)

        def texts(batch):
            ids = self.tokenizer(batch, padding="max_length", max_length=64, truncation=True,
                                 return_tensors="pt")["input_ids"].to(self.device)
            return self.model.get_text_features(input_ids=ids).pooler_output
        self._texts = texts
        self._images = lambda pixels: self.model.get_image_features(
            pixel_values=pixels, interpolate_pos_encoding=self.size != native).pooler_output
        self.torch = torch

    def _load_openai(self):
        import clip
        import torch
        model, _ = clip.load(self.spec["repo"], device="cpu", download_root=str(CACHE / "openai"))
        self.model = model.to(self.dtype).to(self.device)
        self.size = model.visual.input_resolution
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
        self.scale, self.bias = float(model.logit_scale.exp()), 0.
        self._texts = lambda batch: self.model.encode_text(clip.tokenize(batch).to(self.device))
        self._images = self.model.encode_image
        self.torch = torch

    def _load_openclip(self):
        import open_clip
        import torch
        from timm.utils import reparameterize_model
        # The repo's own open_clip_config.json, not open_clip's built-in "MobileCLIP2-B" entry: that one differs
        # (normalisation, and its weights loaded into it embed nonsense: tried, mAP at chance level)
        source = "local-dir:" + str(snapshot(self.spec))
        model, _, preprocess = open_clip.create_model_and_transforms(source)
        model = reparameterize_model(model.eval())          # MobileCLIP's inference form (fused branches)
        self.model = model.to(self.dtype).to(self.device)
        norm = [t for t in preprocess.transforms if type(t).__name__ == "Normalize"][0]
        self.mean, self.std = np.array(norm.mean, np.float32), np.array(norm.std, np.float32)
        size = [t for t in preprocess.transforms if type(t).__name__ == "Resize"][0].size
        self.size = size if isinstance(size, int) else size[0]
        tokenizer = open_clip.get_tokenizer(source)
        self.scale, self.bias = float(model.logit_scale.exp()), float(getattr(model, "logit_bias", None) or 0.)
        self._texts = lambda batch: self.model.encode_text(tokenizer(batch).to(self.device))
        self._images = self.model.encode_image
        self.torch = torch

    def encode_images(self, squares):
        torch = self.torch
        pixels = (np.stack(squares) - self.mean) / self.std
        with torch.inference_mode():
            x = torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()).to(self.device, self.dtype)
            v = self._images(x).float()
            return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()

    def encode_texts(self, groups):
        """[[prompt]] -> one unit vector per group (the normalised mean of its prompts)."""
        torch = self.torch
        flat = [p for g in groups for p in g]
        with torch.inference_mode():
            v = torch.nn.functional.normalize(self._texts(flat).float(), dim=-1).cpu().numpy()
        out, i = [], 0
        for g in groups:
            m = v[i:i + len(g)].mean(0)
            out.append(m / np.linalg.norm(m))
            i += len(g)
        return np.stack(out).astype(np.float32)


# ---- collecting crops --------------------------------------------------------------------------------------
def save_crop(folder, index, crop):
    from PIL import Image
    path = folder / f"{index:05d}.png"
    Image.fromarray(crop).save(path)
    return str(path.relative_to(CROPS))


def box_with_margin(x0, y0, x1, y1, w, h, margin=.1):
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(math.ceil(x1 + mx))), min(h, int(math.ceil(y1 + my))))


def sim_views(scene, desc, rng, per_object):
    """Robot poses (x, y, yaw) looking at each object from 0.7-3 m, clear of furniture and walls."""
    sys.path.insert(0, str(ROOT / "simulation"))
    if scene == "room":
        import world
        things = [(o.name, pos) for o, pos, _, _ in world.ROOM_OBJECTS if o.label != "person"]
        things += [(short, (x, y)) for short, _, x, y, _ in world.GSO_PLACEMENT]
        big = {"couch": 1.0, "table": .9}

        def free(x, y):
            return max(abs(x), abs(y)) < 3.6 and all(math.dist((x, y), p) > big.get(n, .45) for n, p in things)
    else:
        import house
        things = [(o["name"], o["pos"]) for o in desc["objects"]]
        boxes = [house.footprint(o) for o in desc["objects"]]

        def free(x, y):
            room = house.room_at(desc, x, y)
            return (room is not None and all(house.room_at(desc, x + dx, y + dy) == room
                                              for dx, dy in ((.35, 0), (-.35, 0), (0, .35), (0, -.35)))
                    and all(house._gap(b, x, y) > .4 for b in boxes))
    views = []
    for name, (ox, oy) in things:
        for _ in range(per_object):
            for _ in range(40):
                d, a = rng.uniform(.7, 3.), rng.uniform(-math.pi, math.pi)
                x, y = ox + d * math.cos(a), oy + d * math.sin(a)
                if free(x, y):
                    yaw = math.atan2(oy - y, ox - x) + rng.uniform(-.35, .35)
                    views.append((round(x, 2), round(y, 2), round(yaw, 3)))
                    break
    return views


def collect_sim(items, seeds=(1, 2, 3, 4), per_object=3, cap=5):
    import base64
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "simulation"))
    import house
    from go2.adapters.sim import SimAdapter
    folder = CROPS / "sim"
    folder.mkdir(parents=True, exist_ok=True)
    scenes = [("room", "room", None), ("house-demo", "house", house.house_demo())]
    scenes += [(f"house-{s}", "house", house.random_house(random.Random(s))) for s in seeds]
    rng = random.Random(0)
    for scene_id, scene, desc in scenes:
        sim = SimAdapter(python=sys.executable, scene=scene, headless=True, world=desc)
        sim.connect()
        counts = {}
        try:
            views = sim_views(scene, desc, rng, per_object + (1 if scene == "room" else 0))
            for x, y, yaw in views:
                sim.reset(x, y, yaw)
                time.sleep(.4)                                 # let the body settle so the camera is at height
                raw = sim._rpc("camera", timeout=8.0, full=True)
                h, w = raw["height"], raw["width"]
                rgb = np.frombuffer(base64.b64decode(raw["rgb"]), np.uint8).reshape(h, w, 3)
                index = np.frombuffer(base64.b64decode(raw["index"]), np.uint8).reshape(h, w)
                found = []
                for obj in raw["objects"]:
                    if obj["label"] in SIM_SKIP:
                        continue
                    ys, xs = np.nonzero(index == obj["index"])
                    if len(xs) < 120:
                        continue
                    found.append((obj, xs, ys))
                for obj, xs, ys in found:
                    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                    if min(x1 - x0, y1 - y0) < 16 or counts.get(obj["name"], 0) >= cap:
                        continue
                    box = box_with_margin(x0, y0, x1, y1, w, h)
                    inside = {SIM_LABELS.get(o["label"], o["label"]) for o, ox, oy in found if o is not obj and
                              np.mean((ox >= box[0]) & (ox < box[2]) & (oy >= box[1]) & (oy < box[3])) > .4
                              and len(ox) >= 200}
                    crop = np.ascontiguousarray(rgb[box[1]:box[3], box[0]:box[2]])
                    counts[obj["name"]] = counts.get(obj["name"], 0) + 1
                    items.append(dict(file=save_crop(folder, len(items), crop), source="sim", scene=scene_id,
                                      label=SIM_LABELS.get(obj["label"], obj["label"]), object=obj["name"],
                                      size=[int(box[2] - box[0]), int(box[3] - box[1])], contains=sorted(inside)))
        finally:
            sim.close()
        print(f"{scene_id}: {sum(counts.values())} crops of {len(counts)} objects", flush=True)


def collect_coco(items, images=80, per_class=12, small_share=.3):
    import urllib.request
    from PIL import Image
    data = json.loads((CACHE / "coco" / "instances_val2017.json").read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}
    by_image = {}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        if not a["iscrowd"] and names[a["category_id"]] in COCO_CLASSES and min(w, h) >= 24:
            by_image.setdefault(a["image_id"], []).append(a)
    files = {i["id"]: i for i in data["images"]}
    rng = random.Random(0)
    counts, chosen = {}, []
    order = sorted(by_image, key=lambda i: rng.random())
    while len(chosen) < images and order:
        # the image that adds most crops of classes still short of their quota (people don't count)
        def gain(i):
            return sum(1 for a in by_image[i] if names[a["category_id"]] != "person"
                       and counts.get(names[a["category_id"]], 0) < per_class)
        best = max(order[:400], key=gain)
        if gain(best) == 0:
            break
        order.remove(best)
        chosen.append(best)
        for a in by_image[best]:
            counts[names[a["category_id"]]] = counts.get(names[a["category_id"]], 0) + 1
    folder, raw = CROPS / "coco", CACHE / "coco" / "images"
    folder.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    taken = {}
    for image_id in chosen:
        info = files[image_id]
        path = raw / info["file_name"]
        if not path.exists():
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{info['file_name']}", path)
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        anns = by_image[image_id]
        for a in anns:
            label = names[a["category_id"]]
            if taken.get(label, 0) >= (6 if label == "person" else per_class):
                continue
            taken[label] = taken.get(label, 0) + 1
            x, y, bw, bh = a["bbox"]
            box = box_with_margin(x, y, x + bw, y + bh, w, h)
            inside = set()
            for b in anns:
                if b is a:
                    continue
                bx, by, bbw, bbh = b["bbox"]
                ix = max(0, min(box[2], bx + bbw) - max(box[0], bx)) * max(0, min(box[3], by + bbh) - max(box[1], by))
                if ix > .5 * bbw * bbh:
                    inside.add(COCO_LABELS.get(names[b["category_id"]], names[b["category_id"]]))
            crop = np.ascontiguousarray(rgb[box[1]:box[3], box[0]:box[2]])
            common = dict(source="coco", scene=str(image_id), label=COCO_LABELS.get(label, label),
                          object=str(a["id"]), contains=sorted(inside))
            items.append(dict(file=save_crop(folder, len(items), crop), size=[crop.shape[1], crop.shape[0]], **common))
            if rng.random() < small_share:          # the same thing far away: 32-64 px on its longer side
                side = rng.randint(32, 64)
                f = side / max(crop.shape[:2])
                small = Image.fromarray(crop).resize((max(4, round(crop.shape[1] * f)), max(4, round(crop.shape[0] * f))),
                                                     Image.BILINEAR)
                small = np.asarray(small)
                items.append(dict(file=save_crop(folder, len(items), small), size=[small.shape[1], small.shape[0]],
                                  **dict(common, source="coco-small")))
    print(f"coco: {len(chosen)} images, {sum(1 for i in items if i['source'].startswith('coco'))} crops", flush=True)


def load_crops():
    from PIL import Image
    index = json.loads((CROPS / "index.json").read_text())
    return index, [np.asarray(Image.open(CROPS / item["file"]).convert("RGB")) for item in index]


# ---- one model in one process: memory, latency, embeddings ----------------------------------------------------
def run_one(name, device, dtype, dataset):
    marks = {"start": memory()}
    t = time.perf_counter()
    import torch
    torch.manual_seed(0)
    marks["torch"] = memory()
    kind = CANDIDATES[name.split("@")[0]]["kind"]
    if kind == "siglip":
        import transformers  # noqa: F401
    elif kind == "openai":
        import clip  # noqa: F401
    else:
        import open_clip  # noqa: F401
    # The crops are loaded before the baseline. (The results of 10 November loaded them after "loaded", so
    # their "warm" figures include ~100 MB of crops; docs/visual-memory-models.md quotes "loaded" instead.)
    index, crops = load_crops()
    marks["library"] = memory()
    t = time.perf_counter()
    model = Candidate(name, device, dtype)
    load_s = time.perf_counter() - t
    marks["loaded"] = memory()
    rng = random.Random(1)
    sample = [crops[rng.randrange(len(crops))] for _ in range(32)]

    def images(batch):
        return model.encode_images([square(c, model.size, "squash") for c in batch])

    def timed(fn, repeat):
        for _ in range(3):
            fn()
        times = []
        for _ in range(repeat):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        return round(1000 * float(np.median(times)), 1)
    t = time.perf_counter()
    images(sample[:1])
    model.encode_texts(prompts(["bottle"], "photo"))
    first_s = time.perf_counter() - t
    latency = {f"images_{n}": timed(lambda n=n: images(sample[:n]), 10 if n < 32 else 5) for n in (1, 8, 32)}
    latency["text_1"] = timed(lambda: model.encode_texts(prompts(["something to drink from"], "photo")), 20)
    latency["text_1_ensemble"] = timed(lambda: model.encode_texts(prompts(["something to drink from"], "ensemble")), 20)
    marks["warm"] = memory()
    mps = {}
    if device == "mps":
        mps = {"mps_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
               "mps_driver_mb": torch.mps.driver_allocated_memory() / 2**20}
    params = sum(p.numel() for p in model.model.parameters())
    report = dict(model=name, device=device, dtype=dtype, params_m=round(params / 1e6, 1), size=model.size,
                  load_s=round(load_s, 2), first_call_s=round(first_s, 2), latency_ms=latency, memory=marks,
                  scale=model.scale, bias=model.bias, torch=torch.__version__, **mps)
    if dataset:
        queries = query_list(index)
        phrases = [q[0] for q in queries]
        out = {"scale": model.scale, "bias": model.bias}
        for template in TEMPLATES:
            out[f"text_{template}"] = model.encode_texts(prompts(phrases, template))
            out[f"background_{template}"] = model.encode_texts(prompts(BACKGROUND, template))
        t = time.perf_counter()
        for mode in MODES:
            vectors = [model.encode_images([square(c, model.size, mode) for c in crops[i:i + 32]])
                       for i in range(0, len(crops), 32)]
            out[f"image_{mode}"] = np.concatenate(vectors)
        report["dataset_s"] = round(time.perf_counter() - t, 1)
        RESULTS.mkdir(parents=True, exist_ok=True)
        np.savez(RESULTS / f"{name}-{device}-{dtype}.npz", phrases=np.array(phrases), **out)
    return report


def query_list(index):
    """(phrase, type, fits, ignore) for every query: each label in the data, then QUERIES."""
    labels = sorted({item["label"] for item in index})
    return [(label, "exact", {label}, EXACT_IGNORE.get(label, set())) for label in labels] + QUERIES


# ---- scoring -------------------------------------------------------------------------------------------------
def relevance(index, queries):
    """(n_query, n_crop) int8: 1 fits, 0 doesn't, -1 ignore (a near-miss label, or a crop that also shows
    something that fits, e.g. a dining table with cups on it for "cup")."""
    rel = np.zeros((len(queries), len(index)), np.int8)
    for q, (_, _, fits, ignore) in enumerate(queries):
        for c, item in enumerate(index):
            if item["label"] in fits:
                rel[q, c] = 1
            elif item["label"] in ignore or fits & set(item["contains"]):
                rel[q, c] = -1
    return rel


def average_precision(scores, rel):
    keep = rel >= 0
    s, r = scores[keep], rel[keep]
    if r.sum() == 0:
        return None
    r = r[np.argsort(-s, kind="stable")]
    hits = np.cumsum(r)
    return float((hits / np.arange(1, len(r) + 1) * r).sum() / r.sum())


def precision_at(scores, rel, k=5):
    keep = rel >= 0
    s, r = scores[keep], rel[keep]
    if r.sum() == 0:
        return None
    return float(r[np.argsort(-s, kind="stable")[:k]].sum() / min(k, r.sum()))


def rule_scores(data, template, mode, rule):
    """(n_query, n_crop) scores of one decision rule."""
    text, image, background = data[f"text_{template}"], data[f"image_{mode}"], data[f"background_{template}"]
    cos = text @ image.T
    if rule == "cosine":
        return cos
    if rule == "zscore":     # each phrase against the whole memory (no labels used): how unusual is this crop?
        return (cos - cos.mean(1, keepdims=True)) / cos.std(1, keepdims=True)
    bg = background @ image.T                                       # (n_background, n_crop)
    if rule == "margin":
        return cos - bg.max(0, keepdims=True)
    scale = float(data["scale"])                                     # softmax over [query] + background
    z = np.concatenate([cos[:, None, :], np.broadcast_to(bg[None], (len(cos),) + bg.shape)], 1) * scale
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, 0] / e.sum(1)


def best_threshold(scores, labels, target=None):
    """The threshold with the highest F1 on these (score, fits) pairs, or with target=0.8 the lowest one whose
    precision is still at least 0.8 (the robot acts on a match, so precision may matter more)."""
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    if target is None:
        i = int(np.argmax(2 * tp / (tp + fp + y.sum())))
    else:
        ok = np.nonzero(tp / (tp + fp) >= target)[0]
        i = int(ok[-1]) if len(ok) else 0
    return float(s[i] if i + 1 == len(s) else (s[i] + s[i + 1]) / 2)


def decide(scores, rel, types, threshold=None, folds=2, target=None):
    """Threshold chosen on one half of the crops (F1-best, or for a target precision) and tested on the other
    (and the other way round). Returns precision, recall, F1 overall, recall per query type, and false alarms on
    "none" queries."""
    n = scores.shape[1]
    fold = np.arange(n) % folds
    stats = []
    for k in range(folds):
        fit, test = fold != k, fold == k
        mask_fit = rel[:, fit] >= 0
        t = threshold if threshold is not None else best_threshold(scores[:, fit][mask_fit],
                                                                   rel[:, fit][mask_fit].astype(np.int64), target)
        s, r = scores[:, test], rel[:, test]
        accept = s >= t
        valid = r >= 0
        tp = (accept & (r == 1)).sum()
        fp = (accept & (r == 0)).sum()
        fn = (~accept & (r == 1)).sum()
        row = dict(threshold=t, precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn))
        for kind in ("exact", "synonym", "loose", "related"):
            sel = types == kind
            row[f"recall_{kind}"] = (accept[sel] & (r[sel] == 1)).sum() / max(1, (r[sel] == 1).sum())
            row[f"precision_{kind}"] = (accept[sel] & (r[sel] == 1)).sum() / max(1, (accept[sel] & valid[sel]).sum())
        none = types == "none"
        row["none_false_accept"] = accept[none].mean()                      # share of crops wrongly accepted
        row["none_queries_with_any"] = accept[none].any(1).mean()
        stats.append(row)
    out = {key: float(np.mean([s[key] for s in stats])) for key in stats[0]}
    out["f1"] = 2 * out["precision"] * out["recall"] / max(1e-9, out["precision"] + out["recall"])
    return out


def evaluate(path, index, queries, rel):
    data = dict(np.load(path))
    types = np.array([q[1] for q in queries])
    sources = np.array([item["source"] for item in index])
    out = {}
    for template in TEMPLATES:
        for mode in MODES:
            cos = rule_scores(data, template, mode, "cosine")
            row = {}
            for kind in ("exact", "synonym", "loose", "related"):
                sel = np.nonzero(types == kind)[0]
                aps = [average_precision(cos[q], rel[q]) for q in sel]
                p5 = [precision_at(cos[q], rel[q]) for q in sel]
                row[f"map_{kind}"] = float(np.mean([a for a in aps if a is not None]))
                row[f"p5_{kind}"] = float(np.mean([p for p in p5 if p is not None]))
            for source in ("sim", "coco", "coco-small"):
                cols = sources == source
                sel = np.nonzero(types == "exact")[0]
                aps = [average_precision(cos[q][cols], rel[q][cols]) for q in sel]
                row[f"map_exact_{source}"] = float(np.mean([a for a in aps if a is not None]))
            out[f"{template}/{mode}"] = row
    return data, out


def report(models):
    index, _ = load_crops()
    queries = query_list(index)
    rel = relevance(index, queries)
    types = np.array([q[1] for q in queries])
    summary = {"crops": len(index), "by_source": {s: sum(i["source"] == s for i in index)
                                                  for s in ("sim", "coco", "coco-small")},
               "labels": sorted({i["label"] for i in index}), "queries": len(queries), "models": {}}
    print(f"{len(index)} crops {summary['by_source']}, {len(summary['labels'])} labels, {len(queries)} queries")
    for name in models:
        path = RESULTS / f"{name}-mps-fp32.npz"
        if not path.exists():
            continue
        data, retrieval = evaluate(path, index, queries, rel)
        best = max(retrieval, key=lambda k: retrieval[k]["map_exact"] + retrieval[k]["map_synonym"]
                   + retrieval[k]["map_loose"] + retrieval[k]["map_related"])
        template, mode = best.split("/")
        rules = {rule: decide(rule_scores(data, template, mode, rule), rel, types)
                 for rule in ("cosine", "margin", "softmax", "zscore")}
        rules.update({rule + "@P0.8": decide(rule_scores(data, template, mode, rule), rel, types, target=.8)
                      for rule in ("cosine", "margin", "softmax")})
        if data["bias"] != 0:     # SigLIP's own probability at 0.5, uncalibrated
            prob = 1 / (1 + np.exp(-(rule_scores(data, template, mode, "cosine") * data["scale"] + data["bias"])))
            rules["sigmoid@0.5"] = decide(prob, rel, types, threshold=.5)
            rules["sigmoid@0.1"] = decide(prob, rel, types, threshold=.1)
        timing = {}
        for run in ("mps-fp16", "cpu-fp32", "mps-fp32"):
            p = RESULTS / f"{name}-{run}.json"
            if p.exists():
                timing[run] = json.loads(p.read_text())
        summary["models"][name] = dict(best=best, retrieval=retrieval, rules=rules, timing=timing)
        r = retrieval[best]
        print(f"\n== {name}: best {best}  mAP exact {r['map_exact']:.3f} synonym {r['map_synonym']:.3f} "
              f"loose {r['map_loose']:.3f} related {r['map_related']:.3f} | sim {r['map_exact_sim']:.3f} "
              f"coco {r['map_exact_coco']:.3f} small {r['map_exact_coco-small']:.3f}")
        for rule, s in rules.items():
            print(f"   {rule:12s} t={s['threshold']:.4f} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
                  f"R(ex/syn/loose/rel)={s['recall_exact']:.2f}/{s['recall_synonym']:.2f}/{s['recall_loose']:.2f}/"
                  f"{s['recall_related']:.2f} none-FA={s['none_false_accept']:.4f} none-any={s['none_queries_with_any']:.2f}")
        for key in sorted(retrieval):
            r = retrieval[key]
            print(f"   {key:16s} exact {r['map_exact']:.3f}/{r['p5_exact']:.2f} syn {r['map_synonym']:.3f}/"
                  f"{r['p5_synonym']:.2f} loose {r['map_loose']:.3f}/{r['p5_loose']:.2f} related "
                  f"{r['map_related']:.3f}/{r['p5_related']:.2f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return summary


def verify(model, device):
    """The shipped go2.embedder itself, in this (fresh) process: memory, latency, and its retrieval and decision
    rule on the benchmark crops. The GPU is woken first, as YOLO will have done in the robot's process, so the
    memory added is the model's own."""
    sys.path.insert(0, str(ROOT))
    marks = {"start": memory()}
    import torch
    from go2.embedder import Embedder
    import transformers  # noqa: F401
    if device == "mps":
        (torch.ones(64, 64, device="mps") @ torch.ones(64, 64, device="mps")).cpu()
    index, crops = load_crops()                 # before the baseline: the crops themselves are ~100 MB
    marks["libraries"] = memory()
    embedder = Embedder(model, device)
    t = time.perf_counter()
    embedder.load()
    load_s = time.perf_counter() - t
    marks["loaded"] = memory()
    rng = random.Random(1)
    sample = [crops[rng.randrange(len(crops))] for _ in range(32)]
    words = iter(f"thing number {i}" for i in range(10_000))      # new phrases: the text cache must not help

    def timed(fn, repeat):
        for _ in range(3):
            fn()
        times = []
        for _ in range(repeat):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        return round(1000 * float(np.median(times)), 1)
    latency = {f"images_{n}": timed(lambda n=n: embedder.images(sample[:n]), 10) for n in (1, 8, 32)}
    latency["text_1"] = timed(lambda: embedder.texts([next(words)]), 20)
    latency["text_1_cached"] = timed(lambda: embedder.texts(["bottle"]), 20)
    marks["warm"] = memory()
    t = time.perf_counter()
    image = np.concatenate([embedder.images(crops[i:i + 64]) for i in range(0, len(crops), 64)])
    queries = query_list(index)
    text = embedder.texts([q[0] for q in queries])
    dataset_s = time.perf_counter() - t
    marks["after_dataset"] = memory()
    rel = relevance(index, queries)
    types = np.array([q[1] for q in queries])
    cos = text @ image.T
    score = embedder.match(text, image)
    retrieval = {}
    for kind in ("exact", "synonym", "loose", "related"):
        sel = np.nonzero(types == kind)[0]
        retrieval[f"map_{kind}"] = float(np.mean([a for a in (average_precision(cos[q], rel[q]) for q in sel)
                                                  if a is not None]))
    rules = {"threshold": decide(score, rel, types, threshold=embedder.threshold),
             "strict_threshold": decide(score, rel, types, threshold=embedder.strict_threshold)}
    mps = {}
    if embedder.device == "mps":
        mps = {"mps_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
               "mps_driver_mb": torch.mps.driver_allocated_memory() / 2**20}
    reference = RESULTS / f"{model.replace('-192', '@192')}-mps-fp32.npz"
    agreement = None
    if reference.exists() and "@" not in model:   # same crops, same preprocessing: how close to the benchmark run?
        ref = np.load(reference)["image_pad"]
        agreement = float(np.min(np.sum(ref * image, 1)))
    return dict(model=embedder.name, device=embedder.device, threshold=embedder.threshold,
                strict_threshold=embedder.strict_threshold, load_s=round(load_s, 2), latency_ms=latency,
                dataset_s=round(dataset_s, 1), memory=marks, retrieval=retrieval, rules=rules,
                min_cosine_to_benchmark_vectors=agreement, **mps)


def python_for(name):
    return SIDE_VENV if CANDIDATES[name.split("@")[0]]["kind"] == "openclip" else Path(sys.executable)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    sub.add_parser("collect")
    e = sub.add_parser("embed")
    e.add_argument("models", nargs="*", default=list(CANDIDATES))
    e.add_argument("--runs", nargs="+", default=["mps-fp32+data", "mps-fp16", "cpu-fp32"])
    one = sub.add_parser("one")                 # internal: a single fresh process
    one.add_argument("model")
    one.add_argument("device")
    one.add_argument("dtype")
    one.add_argument("--data", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("models", nargs="*", default=list(CANDIDATES))
    v = sub.add_parser("verify", help="measure the shipped go2/embedder.py (one fresh process per run)")
    v.add_argument("model", nargs="?", default="siglip2-b16")
    v.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.command == "fetch":
        from huggingface_hub import snapshot_download
        for name, spec in CANDIDATES.items():
            if spec["kind"] == "openai":
                import clip
                clip.load(spec["repo"], device="cpu", download_root=str(CACHE / "openai"))
            else:
                snapshot_download(spec["repo"], revision=spec["revision"], cache_dir=CACHE / "hf", allow_patterns=HF_FILES)
            print("have", name)
    elif args.command == "collect":
        items = []
        collect_sim(items)
        collect_coco(items)
        (CROPS / "index.json").write_text(json.dumps(items, indent=0) + "\n")
        print(len(items), "crops")
    elif args.command == "one":
        result = run_one(args.model, args.device, args.dtype, args.data)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{args.model}-{args.device}-{args.dtype}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result))
    elif args.command == "embed":
        for name in args.models:
            python = python_for(name)
            if not python.exists():
                print(f"skip {name}: {python} missing (see the module docstring)")
                continue
            for run in args.runs:
                device, dtype = run.split("+")[0].split("-")
                command = [str(python), "-I", str(Path(__file__).resolve()), "one", name, device, dtype]
                command += ["--data"] if run.endswith("+data") else []
                done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                tail = (done.stdout.strip().splitlines() or [""])[-1]
                print(name, run, tail[:400] if done.returncode == 0 else done.stderr[-1500:], flush=True)
    elif args.command == "report":
        report(args.models)
    elif args.command == "verify":
        result = verify(args.model, args.device)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"verify-{args.model}-{args.device}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1))
    print(platform.platform())


if __name__ == "__main__":
    main()
