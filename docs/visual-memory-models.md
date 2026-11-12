# Semantic object memory: which image-text embedding model

The semantic object memory (`go2/visual_memory.py`) stores a vector for every object crop the robot sees, with where and when. Later a phrase
("bottle", "something to drink from", "my shoes") finds the crops that fit, including things no longer in view.
`go2/embedder.py` turns crops and phrases into those vectors and decides whether a crop fits a phrase. This page
says which model it uses and why, what was measured on the target Mac, and what was not.

Measured on 10 November 2026 on the target Mac (Apple M1 Pro, 16 GB, macOS 27.2, Python 3.12.14, torch 2.8.0,
transformers 5.22.2). Latencies are medians. "Memory" is the process's macOS physical footprint (what Activity
Monitor shows as Memory, including GPU buffers), read with `proc_pid_rusage`. RSS is also given where it differs:
it counts memory-mapped weight files, which cost no private memory and can be dropped by the OS.

## The pick

**Google SigLIP 2 ViT-B/16 at 224 px** (`google/siglip2-base-patch16-224`, **Apache-2.0**), through Hugging
Face transformers. It is already installed in the project; no new package is needed. **Fallback: SigLIP ViT-B/16**
(`google/siglip-base-patch16-224`, Apache-2.0), with the same code path: `Embedder("siglip-b16")`.

**Why.**
- **Retrieval.** Of the permissively licensed models measured, SigLIP 2 B/16 retrieved best:
  - exact labels: mAP 0.772, against 0.728 for SigLIP and 0.589 for OpenAI CLIP B/16;
  - synonyms, loose phrases and related things: mAP 0.743, 0.743 and 0.703.
  - It was the only one that kept cups and wine glasses near the top for "bottle" (AP 0.63, against 0.48 for
    SigLIP).
- **A usable yes/no.** SigLIP models are trained with a sigmoid per (image, text) pair. So one fixed cosine
  threshold separates matches from non-matches about as well as any rule tried here.
- **Fit.** It fits the memory and latency budget once its text tower is kept on the CPU (see Memory below).
- **Why not MobileCLIP2-B.** It is better still on exact labels (mAP 0.835, +0.063, significant), but not on
  loose phrases (+0.024, within noise). Its yes/no rule is about as good (F1 0.627 against 0.612). Its weights
  are research-only (below). The gap isn't large enough to pay that licence cost.

| | |
| --- | --- |
| Use | `e = Embedder()`, then `e.images(crops)`, `e.texts(phrases)` and `e.match(t, i) >= e.threshold` |
| Vectors | 768-d, float32, unit length; storing them as float16 changes no decision |
| Decision rule | `match = sigmoid(104.6 * (cosine - 0.0926))`, a match when `>= threshold = 0.5` |
| Stricter rule | `e.strict_threshold = 0.778`, the cosine where precision reached 0.8 (0.1046) |
| Prompt | The bare phrase, lowercased. Templates such as "a photo of a ..." were 1-3 points worse for SigLIP 2 |
| Crop | Centred on a mid-grey square, then bicubic to 224 px. Padding beat stretching by 3 points and centre-cutting by 9 |
| Weights | `python -m go2.embedder fetch`: 5 files, 1.54 GB, into `models/siglip2-base-patch16-224/` (git-ignored), pinned to commit `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` and SHA-256 (`model.safetensors` `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`) |
| Offline | After fetching, nothing downloads. `available()` only looks at the files; the first load checks the SHA-256 once and records the result in `.verified.json` |
| Image latency | 8 crops in 106-111 ms on MPS (p90 114 ms); 141 ms while YOLO11n-seg runs flat out on the same GPU; 1 crop 27 ms |
| Text latency | A new phrase in 30 ms (CPU); a repeated phrase is cached |
| Load | 2.2 s |
| Memory | +487 MB after load, +572 MB steady (standalone, after 400 looks: no growth). +265 MB in a process that already runs YOLO on the GPU |

**The rule's slope was not imposed.** It is a logistic fit of "fits" against cosine over 77,000 labelled
(phrase, crop) pairs. It came out at 104.6, within 10% of SigLIP 2's own logit scale of 112.7. So the model's
sigmoid has the right shape and only its offset was wrong. The model's own bias (-16.77) puts 0.5 at a cosine of
0.149. That is so conservative that it accepted 1% of the true matches here (precision 1.00, recall 0.010).

**Faster variant.** `Embedder("siglip2-b16-192")` uses the same weights at 192 px, with the position embeddings
interpolated.
- It takes 82 ms for 8 crops instead of 106-111 ms.
- mAP is 1.4 points lower on exact labels and 1.1 lower on the rest, both within noise.
- Its vectors are not comparable with 224 px ones: the store should keep `Embedder.name` with them.

## Results on this Mac

### Retrieval

1,001 labelled crops and 77 queries (method below). Each model is shown at its best prompt and crop setting.
The mAP columns are by query type, all queries. The sim, COCO and small columns are exact-label mAP within each
part of the data. P@5 is the share of the top 5 that fit (exact labels).

| Model (licence) | Setting | mAP exact (52) | synonym (8) | loose (8) | related (2) | sim | COCO | small | P@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SigLIP 2 B/16 224** (Apache-2.0) | bare phrase, pad | **0.772** | **0.743** | **0.743** | **0.703** | 0.786 | 0.784 | 0.714 | 0.92 |
| SigLIP 2 B/16 at 192 px | bare phrase, pad | 0.758 | 0.735 | 0.740 | 0.649 | 0.801 | 0.756 | 0.702 | 0.91 |
| SigLIP 2 B/16 at 160 px | bare phrase, pad | 0.736 | 0.716 | 0.722 | 0.656 | 0.780 | 0.747 | 0.657 | 0.89 |
| SigLIP B/16 224 (Apache-2.0) | "this is a photo of …", pad | 0.728 | 0.715 | 0.689 | 0.570 | 0.820 | 0.737 | 0.694 | 0.84 |
| SigLIP B/16 at 160 px | same | 0.661 | 0.674 | 0.681 | 0.558 | 0.776 | 0.662 | 0.631 | 0.83 |
| MobileCLIP2-B (Apple, research only) | bare phrase, pad | 0.835 | 0.799 | 0.746 | 0.683 | 0.845 | 0.825 | 0.823 | 0.96 |
| OpenAI CLIP B/16 (MIT) | bare phrase, pad | 0.589 | 0.596 | 0.617 | 0.655 | 0.673 | 0.678 | 0.608 | 0.76 |

**Paired differences in mAP, per query.** The intervals are 95% bootstrap intervals over the queries.

| Comparison | Exact labels | Synonym, loose and related |
| --- | --- | --- |
| SigLIP 2 minus SigLIP | +0.043 (+0.019 to +0.066) | +0.052 (-0.002 to +0.104) |
| MobileCLIP2-B minus SigLIP 2 | +0.063 (+0.030 to +0.104) | +0.024 (-0.032 to +0.084) |
| SigLIP 2 minus OpenAI CLIP | +0.183 (+0.127 to +0.246) | +0.127 (+0.055 to +0.200) |
| SigLIP 2 at 224 px minus 192 px | +0.014 (-0.009 to +0.041) | +0.011 (-0.016 to +0.036) |

The sources disagree in one place. SigLIP (first version) did best on the rendered simulator crops (0.820), and
SigLIP 2 on real photos and small crops. The robot's real camera is closer to the photos.

**Per query (AP).** Columns are SigLIP 2 / SigLIP / MobileCLIP2-B / CLIP.

| Type | Query | SigLIP 2 / SigLIP / MobileCLIP2-B / CLIP |
| --- | --- | --- |
| Synonym | sofa | 0.81 / 0.77 / 0.82 / 0.74 |
| Synonym | mug | 0.82 / 0.70 / 0.84 / 0.61 |
| Synonym | television | 0.71 / 0.70 / 0.81 / 0.41 |
| Synonym | fridge | 0.43 / 0.47 / 0.54 / 0.37 |
| Synonym | plant | 0.91 / 0.83 / 0.96 / 0.87 |
| Synonym | table | 0.42 / 0.47 / 0.61 / 0.46 |
| Synonym | stuffed animal | 0.93 / 0.89 / 0.93 / 0.83 |
| Synonym | bag | 0.92 / 0.89 / 0.89 / 0.49 |
| Loose | something to drink from | 0.74 / 0.55 / 0.73 / 0.87 |
| Loose | a toy | 0.82 / 0.57 / 0.86 / 0.55 |
| Loose | somewhere to sit | 0.86 / 0.67 / 0.82 / 0.69 |
| Loose | food | 0.80 / 0.78 / 0.67 / 0.72 |
| Loose | a screen | 0.77 / 0.78 / 0.55 / 0.49 |
| Loose | something to read | 0.73 / 0.74 / 0.71 / 0.88 |
| Loose | a pet | 0.56 / 0.79 / 0.94 / 0.13 |
| Loose | kitchen appliance | 0.66 / 0.64 / 0.70 / 0.60 |
| Related | bottle (cups and glasses count) | 0.63 / 0.48 / 0.53 / 0.50 |
| Related | a drink | 0.78 / 0.66 / 0.83 / 0.81 |

**Prompts.**
- Every model was tried with:
  - the bare phrase;
  - "a photo of …";
  - SigLIP's recommended "this is a photo of …";
  - an average over six templates ("… on the floor", "a low-angle photo of …", …).
- The six-template average was the worst for all four models, 6-8 points below the best.

**Crops.**
- Padding to a square beat stretching (SigLIP's own preprocessing) and centre-cutting (CLIP's) for every model.
  For SigLIP 2, exact-label mAP was 0.772 padded, 0.740 stretched and 0.684 cut.
- A stretched couch or a cut-off bottle is harder to recognise.

### One yes/no rule across all phrases

**How the rules were scored.**
- For every rule, the threshold with the best F1 was chosen on half the crops and tested on the other half, and
  then the other way round.
- The pairs scored are all (query, crop) pairs except the ignored ones (method below), about 77,000.
- "Nothing-fits" queries ask for something absent from the data: rubbish bin, my shoes, a guitar, car keys, a fire
  extinguisher, a vacuum cleaner, a bicycle.

**The rules compared.**
- **Cosine:** a threshold on the cosine. This is the same as SigLIP's sigmoid with its bias re-fitted.
- **Softmax:** the phrase against six background prompts ("an object", "the floor", "a wall", …) at the model's
  temperature.
- **Margin:** the phrase's cosine minus the best background prompt's cosine.
- **z-score:** each phrase's cosine standardised over the whole memory.

| Model | Rule | Precision | Recall | F1 | Recall by type: exact / synonym / loose / related | Nothing-fits: crops accepted |
| --- | --- | --- | --- | --- | --- | --- |
| **SigLIP 2** | **cosine** (threshold 0.093) | 0.63 | 0.59 | **0.612** | 0.72 / 0.67 / 0.39 / 0.38 | 0.33% |
| SigLIP 2 | softmax / margin / z-score | | | 0.581 / 0.567 / 0.561 | | 0.23% / 0.29% / 1.3% |
| SigLIP 2 | cosine, threshold for precision 0.8 | 0.80 | 0.45 | 0.573 | 0.61 / 0.52 / 0.20 / 0.15 | 0.11% |
| SigLIP 2 | its own sigmoid at 0.5 | 1.00 | 0.010 | 0.02 | | 0 |
| SigLIP | softmax / margin / cosine | 0.61 | 0.59 | 0.599 / 0.598 / 0.588 | 0.68 / 0.71 / 0.43 / 0.38 (softmax) | 0.24% |
| SigLIP | its own sigmoid at 0.5 | 0.94 | 0.046 | 0.09 | | 0 |
| MobileCLIP2-B | cosine | 0.70 | 0.57 | 0.627 | 0.71 / 0.59 / 0.40 / 0.29 | 0.40% |
| OpenAI CLIP B/16 | softmax / cosine | 0.56 | 0.49 | 0.521 / 0.443 | | 0.51% |

**What to expect from the shipped rule** (score >= 0.5, all crops):
- **Overall:** precision 0.62, recall 0.61.
- **By query type:** exact labels P 0.57 / R 0.72; synonyms 0.78 / 0.68; loose phrases 0.72 / 0.42; related
  things 0.80 / 0.39. Synonym and loose phrases are accepted less often but are right more often.
- **Nothing-fits queries:** 0.36% of crops accepted. Six of the seven queries accepted at least one of the 1,001
  crops:

  | Query | Crops accepted | What they were |
  | --- | --- | --- |
  | rubbish bin | 10 | bowls, cups, a sink, a toaster, a toilet |
  | my shoes | 6 | backpack, suitcase, toy dog, … |
  | car keys | 4 | backpack, mouse, scissors |
  | a fire extinguisher | 2 | |
  | a vacuum cleaner | 2 | |
  | a guitar | 1 | |
  | a bicycle | 0 | |

- **With `strict_threshold` (0.778):** precision 0.80 and recall 0.45. Only "rubbish bin" (6 cups), "my shoes"
  (1 backpack) and "a fire extinguisher" (1) accepted anything.
- **What this means for the robot.** Rank by the score, act on matches at or above 0.5, and go and look before
  saying "found it". Use the strict threshold when acting on a match is costly.

### Latency and memory

**Benchmark harness.** Every candidate loaded the plain way (the whole model on one device), each in a fresh
process. Times are in ms.

| Model | Params | MPS fp16: 1 / 8 / 32 crops, text | CPU fp32: 1 / 8 / 32 crops, text | Weights on GPU (fp16) | Footprint after a plain fp16 load on MPS |
| --- | --- | --- | --- | --- | --- |
| SigLIP 2 B/16 | 375 M | 31 / 119 / 416, 13 | 58 / 249 / 1011, 31 | 716 MB | +1,329 MB |
| SigLIP 2 B/16 at 192 px | 375 M | 26 / 92 / 306, 13 | 55 / 187 / 812, 39 | 716 MB | +1,327 MB |
| SigLIP B/16 | 203 M | 31 / 119 / 417, 13 | 60 / 245 / 1056, 32 | 387 MB | +1,584 MB |
| MobileCLIP2-B | 150 M | 25 / 108 / 378, 12 | 55 / 230 / 1028, 26 | 286 MB | +1,693 MB |
| OpenAI CLIP B/16 | 150 M | 26 / 133 / 478, 11 | 54 / 256 / 1057, 27 | 289 MB | +1,901 MB |

- **The image towers are all about the same cost.** Every candidate is a ViT-B/16 or similar at 224 px, so 8 crops
  take 110-135 ms on the M1 Pro's GPU, whatever the model.
- **Half precision on MPS barely helps.** fp32 was within 10%.
- **Why a plain load costs so much more than the weights.** On Apple Silicon, PyTorch's MPS allocator puts any
  buffer between 10 and 512 MB into a 1 GiB Metal heap.
- **What `go2/embedder.py` therefore does:**
  - It keeps the text tower (SigLIP 2's is mostly a 256k-word embedding table) on the CPU in float32, straight from
    the memory-mapped weights file.
  - It moves only the image tower to the GPU in float16. Every parameter there is under 10 MB.
  - It embeds at most 8 crops per GPU pass, so the activations stay under 10 MB too.
- **Measured with the shipped module:**

  | | Footprint added | Detail |
  | --- | --- | --- |
  | MPS, standalone | +487 MB after load, +572 MB steady | 400 looks of 8 crops plus a query every 5 looks: no growth after the first 100. GPU memory: 177 MB allocated, 265 MB reserved. SigLIP 2's tokenizer is 163 of these MB; SigLIP's is a few MB |
  | MPS, in a process that already runs YOLO11n-seg | +265 MB | The GPU heaps are shared |
  | CPU only | +298 MB after load, +385 MB steady | 8 crops take 273-295 ms |

- **Reproducibility.** The vectors match the benchmark's float32 ones to a cosine of at least 0.99983. mAP and
  precision/recall are identical to the third decimal. `verify` and its JSON give the details.

## Method

**Simulator crops (427).**
- **Scenes:**
  - `SimAdapter(scene="room", headless=True)`, whose household objects are Google Scanned Objects;
  - the demo house (`house_demo()`);
  - `random_house` with seeds 1-4. The realistic furniture download was present.
- **Views.** For every object, three or four robot poses (`reset(x, y, yaw)`) 0.7-3 m away, looking at it with up to
  ±20° of error. Poses clear of furniture and walls.
- **Crops.** The full 1280x720 camera with ground-truth instance masks (`_rpc("camera", full=True)`; `camera()`
  returns half size). Each visible object's mask bounding box with a 10% margin, at least 16 px on the shorter side.
  At most 5 crops per object.
- **Result.** 25 labels. Longest side: median 191 px, 5-95% 35-798 px.

**COCO val2017 crops (440 + 134 small).**
- **Images.** The official instance annotations. 80 images chosen greedily to cover 44 household classes. Boxes at
  least 24 px, not crowd annotations, at most 12 per class (6 for people).
- **Small copies.** 30% of the crops were also shrunk to 32-64 px on the longer side, as far-away objects. These
  134 are the "small" column.
- **Not committed.** Images and annotations stay in `_deps/embedding-cache/coco/`.

**Queries (77).**
- **52 exact:** every label in the data, as the detector names it ("tv", "dining table").
- **8 synonyms:** sofa, mug, television, fridge, plant, table, stuffed animal, bag.
- **8 loose phrases:** something to drink from, a toy, somewhere to sit, food, a screen, something to read, a pet,
  kitchen appliance.
- **2 related:** "bottle" and "a drink", with cups and wine glasses counting as fits.
- **7 nothing-fits.**

**What counts as a fit.**
- Each query lists the labels that fit it, and near-misses that count neither way. Examples of near-misses: bed and
  toilet for "somewhere to sit"; real dogs for "toy dog".
- A crop is also ignored for a query when it contains something that fits: a dining table with cups on it, for
  "cup". For COCO this means at least half of that object's box is inside the crop; for the simulator, 40% of its
  mask.

**Not available.** There are no cans in either set. The "bottle should bring back cans" case could only be tested
as "bottle brings back cups and glasses".

**Metrics.**
- mAP and P@5 per query type.
- Paired bootstrap over queries for the differences between models.
- Decision rules as described above, with 2-fold cross-validation over crops.

**Fresh processes.** Memory, latency and embeddings come from a fresh process per model and device. MobileCLIP2
ran from a separate venv with `open_clip_torch` 4.2.1 and `timm` 1.0.31 (under `_deps/`, not the project's).

**A trap worth knowing.** Loading MobileCLIP2-B as `open_clip.create_model_and_transforms("MobileCLIP2-B",
pretrained=file)` gives embeddings at chance level (mAP 0.05). The built-in config's normalisation differs from
the repo's. The repo's own `open_clip_config.json` (`local-dir:`) works, and is what was used.

**Tokenizer without sentencepiece.**
- `PreTrainedTokenizerFast` reads the repo's `tokenizer.json`.
- In the throwaway venv, it gave exactly the same token ids as the sentencepiece tokenizers (SigLIP's
  `SiglipTokenizer`, SigLIP 2's `GemmaTokenizer`). That held for all 380 prompts tried.
- SigLIP 2's tokenizer does not lowercase, so `texts()` does.

## Candidates surveyed

Sizes and licences of the measured and Hugging Face-hosted models were read from the Hugging Face API at the
pinned commits. Published numbers come from the cited pages: ImageNet zero-shot top-1 (IN), and COCO text-to-image
recall@1 where given. The rest was compiled from model cards and papers in November 2026. Entries marked
"unconfirmed" could not be checked.

| Model | Params (image + text) | Download | Dim | Weight licence | IN | COCO T→I R@1 | Runs here with | Measured? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | 151 M | 605 MB | 512 | MIT ([repo](https://github.com/openai/CLIP/blob/main/LICENSE)); card: deployment "out of scope" | 63.3 | | `clip`, `CLIPModel` | no (B/16 instead) |
| [OpenAI CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16) | 150 M | 351 MB (.pt) | 512 | MIT, same card caveat | 68.3 | | `clip` | **yes** |
| [LAION ViT-B/32 laion2B](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) | 151 M | 605 MB | 512 | MIT | 66.6 | | `CLIPModel` | no |
| [LAION ViT-B/16 DataComp-XL](https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K) | 150 M | 599 MB | 512 | MIT | 73.5 | | open_clip only | no |
| [Apple DFN2B ViT-B/16](https://huggingface.co/apple/DFN2B-CLIP-ViT-B-16) | 150 M | 599 MB | 512 | Apple research-only | 76.2 | | open_clip only | no |
| [SigLIP B/16-224](https://huggingface.co/google/siglip-base-patch16-224) | 203 M | 813 MB | 768 | Apache-2.0 | 76.2 | 47.2 | `SiglipModel` + tokenizer.json | **yes (fallback)** |
| [SigLIP 2 B/16-224](https://huggingface.co/google/siglip2-base-patch16-224) | 375 M (86 + 282; 197 M of it the 256k-word embedding table) | 1.50 GB | 768 | Apache-2.0 | 78.2 | 52.1 | `AutoModel` (siglip) + tokenizer.json | **yes (pick)** |
| [SigLIP 2 B/16-256](https://huggingface.co/google/siglip2-base-patch16-256) | 375 M | 1.50 GB | 768 | Apache-2.0 | 79.1 | 53.2 | same | no |
| [SigLIP 2 B/32-256](https://huggingface.co/google/siglip2-base-patch32-256) | 377 M | 1.51 GB | 768 | Apache-2.0 | 74.0 | 47.2 | same | no (would be ~3x faster) |
| [SigLIP 2 B/16 NaFlex](https://huggingface.co/google/siglip2-base-patch16-naflex) | 375 M | 1.50 GB | 768 | Apache-2.0 | unconfirmed | | `Siglip2Model` | no |
| SigLIP / SigLIP 2 large and so400m | 0.9-1.1 B | 3.5-4.5 GB | 1024-1152 | Apache-2.0 | 82.5-83.4 | | | too big |
| [MobileCLIP S0 / S1 / S2 / B](https://github.com/apple/ml-mobileclip) | 54 / 85 / 99 / 150 M | 216-599 MB | 512 | Apple research-only | 67.8 / 72.6 / 74.4 / 76.8 | | open_clip + timm; ONNX ports for S0, S2 | no |
| [MobileCLIP2 S0 / S2 / B / S3 / S4](https://huggingface.co/apple/MobileCLIP2-B) | 75 / 99 / 150 / 249 / 446 M | 300 MB-1.8 GB | 512 (B) | Apple research-only ([LICENSE](https://huggingface.co/apple/MobileCLIP2-B/blob/main/LICENSE)) | 71.5 / 77.2 / 79.4 / 80.7 / 81.9 | | open_clip 4.2 + timm | **B: yes** |
| [TinyCLIP ViT-39M/16 + Text-19M](https://github.com/wkcn/TinyCLIP) | 83 M | 332 MB | 512 | MIT | 63.5 | | `CLIPModel` | no (weaker than CLIP B/16) |
| [EVA02-CLIP-B/16](https://huggingface.co/QuanSun/EVA-CLIP) | 150 M | 300 MB | 512 | MIT | 74.7 | | open_clip + timm | no |
| [Meta PE-Core T16 / S16 (384 px)](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md) | ~50 / ~60 M | 278 / 349 MB | 512 | Apache-2.0 | 62.1 / 72.7 | 33.0 / 42.6 | open_clip + timm; MLX ports | no |
| [Meta PE-Core B16-224](https://huggingface.co/facebook/PE-Core-B16-224) | ~400 M | 1.79 GB | 1024 | Apache-2.0 | 78.4 | 50.9 | open_clip + timm | no |
| [MetaCLIP 2 worldwide S16 / B16](https://github.com/facebookresearch/MetaCLIP) | 390 / 586 M | 1.6 / 2.3 GB | | CC-BY-NC-4.0 | | | `MetaClip2Model` (transformers 5) | no: non-commercial |
| [Jina CLIP v2](https://huggingface.co/jinaai/jina-clip-v2) | 865 M | 1.73 GB | 1024 (truncatable) | CC-BY-NC-4.0 | | | remote code | no: non-commercial, big |
| [Jina CLIP v1](https://huggingface.co/jinaai/jina-clip-v1) | 223 M | 891 MB | 768 | Apache-2.0 | 59.1 | | remote code | no |
| [Nomic embed vision v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) + text | 93 + 137 M | 372 + 547 MB | 768 | Apache-2.0 | 71.0 | | remote code | no |
| [Google TIPSv2 B/14](https://huggingface.co/google/tipsv2-b14) (2026) | 196 M | 784 MB | 768 | Apache-2.0 | | Flickr T→I 80.0 | `TIPSv2Model`, but only a sentencepiece tokenizer | no: needs sentencepiece |
| [FG-CLIP 2 base](https://huggingface.co/qihoo360/fg-clip2-base) | 384 M | 1.54 GB | | Apache-2.0 | | | remote code | no |

Other sources:
- SigLIP and SigLIP 2 numbers: [SigLIP 2 paper, Table 1](https://arxiv.org/abs/2502.14786).
- CLIP-family zero-shot results: [open_clip's results table](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv).
- SigLIP's prompt template and `padding="max_length"`:
  [transformers' SigLIP docs](https://huggingface.co/docs/transformers/model_doc/siglip).
- SigLIP's sigmoid as a probability: no published calibration study was found. A SigLIP author advises fitting the
  bias on your own data ([discussion](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384/discussions/3)). That is
  what the calibration here does.

**Not measured, and why.**
- Several Apache/MIT models need `open_clip` + `timm` (two more packages, so they were not the first choice): PE-Core,
  EVA02 and the LAION/DataComp B/16. Their published numbers do not beat SigLIP 2 B/16, apart from PE-Core B16,
  which is the same size as SigLIP 2.
- SigLIP 2 B/32 would be about three times faster at 4 points lower ImageNet accuracy. It is worth trying if the
  latency budget tightens. 192 px was measured instead.
- MobileCLIP2-S0/S2 would be much faster image towers, but have the same licence problem as B.
- MetaCLIP 2 and Jina v2 were excluded for their non-commercial licences.

## MobileCLIP2 and its licence

**Technically it is the best model measured.** It is +0.063 mAP on exact labels (significant) and +0.024 on looser
phrases (not significant). Its yes/no rule is about as good (F1 0.627 against 0.612). It is the same speed for 8
crops (108 against 119 ms, plain load), and its fp16 weights on the GPU are 286 MB against 177 MB for SigLIP 2's
image tower.

**Its weights are under Apple's Machine Learning Research Model License: research only.** The licence excludes
"any commercial exploitation, product development or use in any commercial product or service". The cost would be:
- a research-only component in the robot's perception;
- a model that would have to be replaced, and every stored vector re-embedded, before any use beyond research;
- two more packages.

The gain is real on exact labels but not decisive. So the default is SigLIP 2 (Apache-2.0), and switching is a
one-model change behind the same interface if the project ever decides otherwise.

## Limits and what was not verified

- **The real Go2 camera.** Not verified. Its 120° lens, JPEG compression, motion blur and the floor-level view of
  real homes are not in this data. The threshold was fitted on rendered and COCO crops. Expect to re-fit it (the
  `report` step) on a few hundred real crops: only `boundary` should need to move.
- **Cans.** There were none in either data set.
- **People and broad categories.** The threshold was fitted on household objects. On Ultralytics' sample photos, the
  right crop still ranked first for these, but scored below 0.5:
  - "person" scored 0.36 on a man, "a man in a suit" 0.58;
  - "a vehicle" scored 0.03 on a bus, while "bus" scored 0.94.
  Don't use the yes/no rule for people (the memory doesn't store them), and prefer object words.
- **Weak queries.** "Fridge" and "table" are hard (AP 0.43 and 0.42). Many refrigerator and table crops are
  partial, close-up surfaces from a low camera.
- **"Rubbish bin".** It accepts bowls, cups and sinks at the default threshold.
- **Not tried:**
  - concurrency with Whisper or the LLM running (only with YOLO);
  - a fine-tune or linear probe on the robot's own crops.
- **Memory figures.** These are for this Mac and torch 2.8. The MPS heap behaviour is PyTorch's
  (`aten/src/ATen/mps/MPSAllocator.h`) and may change with versions.

## Reproducing

```sh
.venv/bin/python -m go2.embedder fetch                          # the pick, 1.54 GB, pinned
.venv/bin/python -m unittest tests.test_embedder                # 17 tests; model tests skip without weights
.venv/bin/python experiments/embedding_models.py fetch          # all candidates (~3.3 GB) into _deps/embedding-cache
.venv/bin/python experiments/embedding_models.py collect        # ~5 min: simulator + COCO crops
.venv/bin/python experiments/embedding_models.py embed          # one fresh process per model and device
.venv/bin/python experiments/embedding_models.py report         # the tables above
.venv/bin/python experiments/embedding_models.py verify         # the shipped module: memory, latency, quality
```

**MobileCLIP2 venv.** MobileCLIP2 rows need a separate venv:

```sh
_deps/tools/uv venv _deps/embedding-venv
_deps/tools/uv pip install --python _deps/embedding-venv/bin/python torch==2.8.0 torchvision==0.23.0 \
    numpy==2.5.4 pillow psutil open_clip_torch timm
```

The OpenAI CLIP rows need OpenAI's `clip` package (`pip install git+https://github.com/openai/CLIP.git`).
`embed` skips MobileCLIP2 when that venv is missing. The figures above are from the runs of 10 November 2026;
`_deps/embedding-cache/results/` holds their JSON.
