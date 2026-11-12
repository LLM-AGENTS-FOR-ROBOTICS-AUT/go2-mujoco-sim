"""Image-text embeddings for the robot's visual memory: what it saw, found again by words.

Every crop of a thing the robot sees becomes a vector (`images`); a phrase such as "bottle" or "something to
drink from" becomes a vector in the same space (`texts`); `match` scores every phrase against every crop, and a
score at or above `threshold` means "this crop fits this phrase". That last part matters as much as the
ranking: the robot acts on the answer, and a query for something it never saw ("my shoes") should come back
(nearly) empty.

The model is Google's SigLIP 2 ViT-B/16 at 224 px (Apache-2.0), chosen by measurement on the target Mac against
SigLIP, OpenAI CLIP ViT-B/16 and Apple's MobileCLIP2-B: docs/visual-memory-models.md has the numbers and
experiments/embedding_models.py reproduces them. SigLIP models are trained with a sigmoid per (image, text) pair
rather than a softmax over a batch, so one fixed threshold on the cosine means much the same for every phrase;
the calibration below was fitted on simulator and COCO crops (its slope came out within 10% of the model's own
logit scale; only the offset needed moving). SigLIP ViT-B/16 (the first version, same licence, smaller text
vocabulary) is the fallback: Embedder("siglip-b16").

Memory, which is shared with the LLM, Whisper and YOLO: the image tower runs on the GPU (MPS) in float16, in
batches of at most 8 crops so that no single buffer reaches 10 MB (PyTorch reserves larger ones in 1 GiB Metal
heaps). The text tower stays on the CPU in float32, straight from the memory-mapped weights file: SigLIP 2's
text tower is mostly a 256k-word embedding table of which a query touches a few rows, and file-backed pages cost
no private memory. A text query is ~30 ms on the CPU.

Optional: importing this needs only numpy. The weights are
downloaded only by `fetch()` (or `python -m go2.embedder fetch`), into models/<repo name>/ (git-ignored), pinned
to a Hugging Face commit and checked against SHA-256 before first use, so the robot runs offline once they are
there. Loading is lazy and one model is shared between threads behind a lock.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
HF = "https://huggingface.co/{repo}/resolve/{revision}/{file}"
MPS_BATCH = 8            # crops per GPU pass: keeps every activation buffer under PyTorch's 10 MB heap limit


class EmbedderError(RuntimeError):
    pass


SIGLIP2_FILES = {
    "config.json": (253, "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7"),
    "model.safetensors": (1500800904, "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"),
    "tokenizer.json": (34363039, "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
    "tokenizer_config.json": (47164, "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8"),
    "special_tokens_map.json": (636, "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351")}
SIGLIP2 = dict(repo="google/siglip2-base-patch16-224", revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
               folder="siglip2-base-patch16-224", files=SIGLIP2_FILES, licence="Apache-2.0", dim=768, native=224,
               template="{}")      # the bare phrase measured best for SigLIP 2 (a template cost 1-3 points of mAP)

# name -> weights, input size, prompt template and calibration. match = sigmoid(slope * (cosine - boundary)):
# 0.5 at `boundary`, the cosine that gave the best F1 on the benchmark (docs/visual-memory-models.md); `strict` is
# the cosine where precision reached 0.8 there. Different sizes give vectors that must not be mixed: the name
# says which.
MODELS = {
    "siglip2-b16": dict(SIGLIP2, size=224, slope=104.6, boundary=.0926, strict=.1046),
    # 25% faster (92 instead of 120 ms for 8 crops on the M1 Pro), 1.4 points lower mAP: position embeddings
    # interpolated to a 12x12 grid
    "siglip2-b16-192": dict(SIGLIP2, size=192, slope=102.3, boundary=.0883, strict=.1050),
    "siglip-b16": dict(
        repo="google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        folder="siglip-base-patch16-224", licence="Apache-2.0", dim=768, native=224, size=224,
        template="this is a photo of {}.", slope=88.8, boundary=.0710, strict=.0832,
        files={"config.json": (432, "cd85b3d28829722820bcb89a2cfbb4160e55fd359249a3044da724166a8d9688"),
               "model.safetensors": (812672320, "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8"),
               "tokenizer.json": (2399357, "c6e405cb7c670d56636a9402c81023a55bc6c3c53d89cf02b92f5c5005bfe920"),
               "tokenizer_config.json": (711, "d6423dae508cc3a129d22ea443841c111832a1a73125b8f25ea8736951698bcb"),
               "special_tokens_map.json": (409, "2b6a1ff67a27e0df9ac0c7d93250fc0d87431c7b366b3d5669217104f9088a26")}),
}
DEFAULT_MODEL = "siglip2-b16"
DETERMINERS = {"a", "an", "the", "my", "your", "our", "his", "her", "their", "some", "something", "somewhere",
               "anything", "this", "that", "these", "those"}
MASS_NOUNS = {"food", "furniture", "rubbish", "clothing", "laundry", "water"}


def normalise_phrase(phrase):
    """Lowercase, single spaces: SigLIP 2's tokenizer does not lowercase, and the models were trained lowercased."""
    return " ".join(str(phrase).lower().split())


def prompt(phrase, template):
    """The text the model sees. Templates with words around the phrase get an article where English wants one
    ('a cup', 'an apple'; 'my shoes', 'something to drink from' and 'food' stay as they are)."""
    if template == "{}":
        return phrase
    first = phrase.split()[0] if phrase else ""
    if phrase and first not in DETERMINERS and phrase not in MASS_NOUNS:
        phrase = ("an " if phrase[0] in "aeiou" else "a ") + phrase
    return template.format(phrase or "something")


def square(crop, size):
    """HxWx3 uint8 RGB -> size x size x 3 float32 in [0, 1]: centred on a mid-grey square (padding beat
    stretching and centre-cutting by 2-4 points of mAP on the benchmark: a stretched couch or a cut-off bottle is
    harder to recognise), then bicubic, so the same crop always gives the same pixels."""
    from PIL import Image
    crop = np.asarray(crop)
    if crop.ndim == 2:
        crop = np.repeat(crop[..., None], 3, axis=2)
    if crop.ndim != 3 or crop.shape[2] not in (3, 4) or crop.shape[0] < 1 or crop.shape[1] < 1:
        raise ValueError(f"a crop must be an HxWx3 RGB image, not an array of shape {crop.shape}")
    if crop.dtype != np.uint8:
        raise ValueError(f"a crop must be uint8 RGB, not {crop.dtype}")
    img = Image.fromarray(np.ascontiguousarray(crop[..., :3]))
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return np.asarray(canvas.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255


class Embedder:
    """Embedder(model=None, device=None). model: a name in MODELS (default: $GO2_EMBEDDER_MODEL, else
    siglip2-b16). device: "mps", "cpu" or "cuda" for the image tower (default: $GO2_EMBEDDER_DEVICE, else the GPU
    when there is one); the text tower always runs on the CPU. $GO2_EMBEDDER_DIR overrides where the weights are
    (default models/<repo name>/)."""

    TEXT_CACHE = 512                     # phrases kept embedded: queries repeat ("bottle", "my keys")

    def __init__(self, model=None, device=None):
        self.name = model or os.environ.get("GO2_EMBEDDER_MODEL") or DEFAULT_MODEL
        if self.name not in MODELS:
            raise EmbedderError(f"Unknown embedding model '{self.name}'; choose one of {', '.join(MODELS)}.")
        self.spec = MODELS[self.name]
        self.dim = self.spec["dim"]
        self.threshold = 0.5
        self.strict_threshold = round(float(self._calibrate(self.spec["strict"])), 3)   # precision ~0.8
        self.folder = Path(os.environ.get("GO2_EMBEDDER_DIR") or MODELS_DIR / self.spec["folder"])
        self.device = device or os.environ.get("GO2_EMBEDDER_DEVICE") or None
        self.lock = threading.Lock()
        self._model = self._tokenizer = self._torch = None
        self._texts = OrderedDict()

    def __repr__(self):
        return f"Embedder({self.name!r}, loaded={self._model is not None}, device={self.device!r})"

    # ---- setup ---------------------------------------------------------------------------------------------
    def _missing_packages(self):
        """Packages that can't be imported, found without importing them (torch alone takes seconds)."""
        def missing(name):
            try:
                return importlib.util.find_spec(name) is None
            except (ImportError, ValueError):     # e.g. blocked with sys.modules[name] = None
                return True
        return [p for p in ("torch", "transformers", "tokenizers", "safetensors", "PIL") if missing(p)]

    def _missing_files(self):
        """Files absent or of the wrong size (the checksum itself is checked when loading)."""
        return [name for name, (size, _) in self.spec["files"].items()
                if not (self.folder / name).is_file() or (self.folder / name).stat().st_size != size]

    def available(self):
        """None if the model can be used, else why not and what to do. Never raises, never downloads."""
        try:
            if self._model is not None:
                return None
            missing = self._missing_packages()
            if missing:
                return (f"Visual memory needs the Python package(s) {', '.join(missing)}: "
                        "pip install -r requirements-vision.txt")
            files = self._missing_files()
            if files:
                return (f"Visual memory's model ({self.name}) is not downloaded: {', '.join(files)} missing or "
                        f"incomplete in {self.folder}. Download it (about "
                        f"{sum(s for s, _ in self.spec['files'].values()) / 1e9:.1f} GB) with: "
                        "python -m go2.embedder fetch")
            return None
        except Exception as error:      # a broken install must not break the caller's check
            return f"Visual memory can't be used: {error}"

    def fetch(self, progress=None):
        """Download the pinned weights into self.folder, checking each file's SHA-256; files already there and
        correct are kept. progress(done_bytes, total_bytes, file_name) is called as it goes. Returns the folder;
        raises EmbedderError on a network or checksum failure (nothing half-written is left behind)."""
        import urllib.error
        import urllib.request
        self.folder.mkdir(parents=True, exist_ok=True)
        total = sum(size for size, _ in self.spec["files"].values())
        done = 0
        for name, (size, digest) in self.spec["files"].items():
            path = self.folder / name
            if path.is_file() and path.stat().st_size == size and self._verified(path, digest):
                done += size
                continue
            url = HF.format(repo=self.spec["repo"], revision=self.spec["revision"], file=name)
            part = path.with_name(path.name + ".part")
            hasher = hashlib.sha256()
            try:
                with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                        hasher.update(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total, name)
            except (OSError, urllib.error.URLError) as error:
                part.unlink(missing_ok=True)
                raise EmbedderError(f"Could not download {url}: {error}. Check the internet connection and "
                                    "try again.") from None
            if hasher.hexdigest() != digest:
                part.unlink(missing_ok=True)
                raise EmbedderError(f"{name} downloaded from {url} does not match its pinned checksum, so it was "
                                    "not kept. Try again later; if it keeps happening the file has changed upstream.")
            part.replace(path)
            self._remember_verified(path, digest)
        return self.folder

    def _verified(self, path, digest):
        """Is `path` the pinned file? Hashing 1.5 GB takes a few seconds, so a match is remembered (by size and
        modification time) in .verified.json beside the weights, and only a changed file is hashed again."""
        stamp = self._stamps().get(path.name)
        stat = path.stat()
        if stamp == [stat.st_size, stat.st_mtime_ns, digest]:
            return True
        if _sha256(path) != digest:
            return False
        self._remember_verified(path, digest)
        return True

    def _stamps(self):
        try:
            return json.loads((self.folder / ".verified.json").read_text())
        except (OSError, ValueError):
            return {}

    def _remember_verified(self, path, digest):
        stamps = self._stamps()
        stat = path.stat()
        stamps[path.name] = [stat.st_size, stat.st_mtime_ns, digest]
        try:
            (self.folder / ".verified.json").write_text(json.dumps(stamps, indent=1) + "\n")
        except OSError:
            pass                         # read-only folder: it is simply checked again next time

    def _load(self):
        """The model, loaded on first use (the caller holds the lock)."""
        if self._model is not None:
            return self._model
        problem = self.available()
        if problem:
            raise EmbedderError(problem)
        for name, (_, digest) in self.spec["files"].items():
            if not self._verified(self.folder / name, digest):
                raise EmbedderError(f"{self.folder / name} differs from the pinned file. Delete it and run: "
                                    "python -m go2.embedder fetch")
        import torch
        from transformers import AutoModel, PreTrainedTokenizerFast
        from transformers.utils import logging as hf_logging
        if self.device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()      # SigLIP's config carries unused CLIP token ids, which draw a warning
        try:
            # float32, memory-mapped from the safetensors file: the text tower stays like this on the CPU
            model = AutoModel.from_pretrained(self.folder, dtype=torch.float32, local_files_only=True)
            # tokenizer.json is the fast tokenizer: no sentencepiece needed, and it gives the same ids
            # (checked against the sentencepiece tokenizers on 380 prompts)
            self._tokenizer = PreTrainedTokenizerFast.from_pretrained(self.folder, local_files_only=True)
        except Exception as error:
            raise EmbedderError(f"Could not load {self.name} from {self.folder}: {error}") from None
        finally:
            hf_logging.set_verbosity(verbosity)
        model.eval().requires_grad_(False)
        if self.device != "cpu":
            model.vision_model.to(self.device, torch.float16)    # parameter by parameter, each well under 10 MB
        self._image_dtype = torch.float32 if self.device == "cpu" else torch.float16
        self._torch, self._model = torch, model
        return model

    def load(self):
        """Load now rather than on the first call (e.g. while the robot is idle). Raises EmbedderError."""
        with self.lock:
            self._load()
        return self

    # ---- embedding ----------------------------------------------------------------------------------------
    def images(self, crops):
        """[HxWx3 uint8 RGB, any size] -> (n, dim) float32 unit vectors."""
        crops = list(crops)
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        if self._model is None:
            self.load()                  # a missing package or file: EmbedderError before any preprocessing
        size = self.spec["size"]
        pixels = (np.stack([square(c, size) for c in crops]) - .5) / .5      # SigLIP's normalisation
        pixels = np.ascontiguousarray(pixels.transpose(0, 3, 1, 2))
        out = []
        with self.lock:
            model = self._load()
            torch = self._torch
            batch = MPS_BATCH if self.device == "mps" else 32
            with torch.inference_mode():
                for i in range(0, len(pixels), batch):
                    x = torch.from_numpy(pixels[i:i + batch]).to(self.device, self._image_dtype)
                    v = model.get_image_features(pixel_values=x,
                                                 interpolate_pos_encoding=size != self.spec["native"]).pooler_output
                    out.append(v.float().cpu().numpy())
        return _unit(np.concatenate(out))

    def texts(self, phrases):
        """[phrase] -> (n, dim) float32 unit vectors (the phrase put into the model's prompt template)."""
        phrases = [normalise_phrase(p) for p in phrases]
        if not phrases:
            return np.zeros((0, self.dim), np.float32)
        with self.lock:
            todo = list(dict.fromkeys(p for p in phrases if p not in self._texts))
            if todo:
                model = self._load()
                torch = self._torch
                ids = self._tokenizer([prompt(p, self.spec["template"]) for p in todo], padding="max_length",
                                      max_length=64, truncation=True, return_tensors="pt")["input_ids"]
                with torch.inference_mode():     # padded to 64 tokens, without a mask: as SigLIP was trained
                    v = _unit(model.get_text_features(input_ids=ids).pooler_output.float().numpy())
                for p, vector in zip(todo, v):
                    self._texts[p] = vector
            out = np.stack([self._texts[p] for p in phrases])
            for p in phrases:
                self._texts.move_to_end(p)
            while len(self._texts) > self.TEXT_CACHE:
                self._texts.popitem(last=False)
        return out.astype(np.float32)

    def _calibrate(self, cosine):
        return 1 / (1 + np.exp(-self.spec["slope"] * (np.asarray(cosine, np.float32) - self.spec["boundary"])))

    def match(self, text_vectors, image_vectors):
        """(n_text, dim), (n_image, dim) -> (n_text, n_image) scores in [0, 1]; >= self.threshold (0.5) means the
        crop fits the phrase (>= self.strict_threshold: fewer, surer matches). Needs no model, so stored vectors
        (float16 is fine) can be matched without loading it."""
        t = np.asarray(text_vectors, np.float32).reshape(-1, self.dim)
        i = np.asarray(image_vectors, np.float32).reshape(-1, self.dim)
        return self._calibrate(t @ i.T).astype(np.float32)


def _unit(v):
    return (v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    """python -m go2.embedder fetch | check [--model NAME]: download the pinned weights, or say whether the
    visual memory's model can be used."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog="python -m go2.embedder", description=main.__doc__)
    parser.add_argument("command", choices=("fetch", "check"))
    parser.add_argument("--model", choices=list(MODELS), default=None)
    args = parser.parse_args(argv)
    embedder = Embedder(args.model)
    if args.command == "fetch":
        shown = [-1]

        def progress(done, total, name):
            percent = int(100 * done / total)
            if percent != shown[0]:
                shown[0] = percent
                print(f"\r{embedder.name}: {percent}% ({name})", end="", file=sys.stderr, flush=True)
        try:
            embedder.fetch(progress)
        except EmbedderError as error:
            print(f"\n{error}", file=sys.stderr)
            return 1
        print(file=sys.stderr)
    problem = embedder.available()
    print(problem or f"{embedder.name} is ready in {embedder.folder}")
    return 1 if problem else 0


if __name__ == "__main__":
    raise SystemExit(main())
