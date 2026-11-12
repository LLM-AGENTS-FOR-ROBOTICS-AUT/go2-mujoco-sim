"""Semantic object memory: what the things Go2 saw looked like, searchable by words.

The object memory (object_memory.py) remembers things by the detector's label, so it can only be asked for
those exact words: "bottle" never brings back the can the detector called a cup, or the thing it had no
confident name for. Here every thing seen is also kept as an embedding: a picture of it (the crop of the camera
frame inside its box) turned into a vector by an image-text model (embedder.py), saved with where and when it
was seen. A question is turned into a vector by the same model's text side, and the pictures whose vectors
lie close to it are the answer: "bottle" returns the bottles and, a little further down, the cans and cups.
It is retrieval over the robot's own sightings, like a small search index.

It works beside the other two, not instead of them:

  detector (YOLO)          finds a named thing in the frame in front of the robot, with a mask to range it by
  object memory            exact recall by label
  visual memory            loose recall by resemblance, including what is no longer in view

Stored per sighting: position (odometry coordinates), time, the detector's label and colour word, a count,
and the vector (float16). A sighting of the same thing from the same spot updates the entry it already has
rather than adding one, and a thing keeps at most a few views of itself (a teddy bear from the front and from
behind are different pictures, and a phrase may fit one and not the other), so the memory holds things, not
frames. People are never stored. Nothing is kept as an image: only the vectors, which can't be turned back
into pictures in any useful way.

The embedder is given from outside (anything with images(), texts(), match(), threshold, name and dim), so
this module needs only numpy and is tested with a toy one.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

SAME_SPOT_M = .6            # a sighting this close to an entry (plus half the thing's size) may be the same thing
# ... if the detector gave both the same label; or, when one of them has no label, if the two pictures' vectors
# are this alike. (Measured with SigLIP 2 on simulator crops: two pictures of one kind of thing have a cosine
# of 0.78 to 0.97, of different kinds 0.62 to 0.82, so resemblance alone can't tell a mug from the couch
# behind it; the label can.)
SAME_LOOK = .85
SAME_VIEW = .9              # two pictures of one thing this alike are the same view of it; less, and the newer one
MAX_VIEWS = 4               # is kept as another view, up to this many a thing (then the oldest view goes)
MAX_ENTRIES = 3000          # the oldest sightings are dropped beyond this (about 4.6 MB of vectors at 768 numbers)
KEEP_VIEW = .8              # a new picture replaces the kept one unless it is much smaller (under this share of
                            # its pixels): the close-up from a metre away says more than the glimpse from across
                            # the room that came after it
STALE_VIEW_S = 24 * 3600.   # ... but a picture this old is replaced by whatever is seen now
SAVE_EVERY_S = 30.          # the file is rewritten at most this often (and when the executor closes)
MIN_CROP_PX = 16            # smaller crops are a few blurred pixels: not worth a vector
CROP_MARGIN = .12           # a little of the surroundings round the box helps the model see what it is
TEXT_CACHE = 64


def crop(picture, box, margin=CROP_MARGIN):
    """The part of an HxWx3 picture inside `box` ([left, top, right, bottom] as fractions), with a margin; None
    if that is too small to be worth embedding."""
    h, w = picture.shape[:2]
    left, top, right, bottom = box
    dx, dy = (right - left) * margin, (bottom - top) * margin
    x0, x1 = max(0, int((left - dx) * w)), min(w, int(np.ceil((right + dx) * w)))
    y0, y1 = max(0, int((top - dy) * h)), min(h, int(np.ceil((bottom + dy) * h)))
    if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
        return None
    return np.ascontiguousarray(picture[y0:y1, x0:x1])


class VisualMemory:
    def __init__(self, path=None, embedder=None, clock=time.time):
        self.path = Path(path) if path is not None else None
        self.embedder = embedder
        self.clock = clock
        self.lock = threading.RLock()
        self.entries = []                       # dicts: id, t, first, x, y, label, color, size_m, seen
        self.vectors = np.zeros((0, self.dim), np.float16)
        self._next = 1
        self._texts = {}                        # phrase -> vector, most recent TEXT_CACHE
        self._dirty, self._saved = False, 0.
        self.load()

    @property
    def dim(self):
        return int(getattr(self.embedder, "dim", 0) or 0)

    @property
    def model(self):
        return str(getattr(self.embedder, "name", "none"))

    def __len__(self):
        return len(self.entries)

    # ---- storage -------------------------------------------------------------------------------
    def _vector_file(self):
        return self.path.with_suffix(".npy")

    def load(self):
        """Read the saved sightings. Vectors from another model (or another size) can't be compared with this
        one's, so a file written with a different embedder is set aside (renamed *.old), not mixed in."""
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            vectors = np.load(self._vector_file())
        except (OSError, ValueError):
            return
        entries = data.get("entries") if isinstance(data, dict) else None
        if (not isinstance(entries, list) or data.get("model") != self.model or vectors.ndim != 2
                or vectors.shape != (len(entries), self.dim)):
            for file in (self.path, self._vector_file()):
                try:
                    file.replace(file.with_name(file.name + ".old"))
                except OSError:
                    pass
            return
        self.entries, self.vectors = entries, vectors.astype(np.float16)
        numbers = [int(e["id"][1:]) for e in entries if str(e.get("id", ""))[1:].isdigit()]
        self._next = max(numbers, default=0) + 1

    def save(self, force=False):
        if self.path is None or not self._dirty or not (force or time.monotonic() - self._saved >= SAVE_EVERY_S):
            return
        with self.lock:
            entries, vectors = list(self.entries), self.vectors.copy()
            self._dirty, self._saved = False, time.monotonic()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps({"model": self.model, "dim": self.dim, "entries": entries}), encoding="utf-8")
            vector_temporary = self.path.with_name(self.path.stem + ".tmp.npy")
            np.save(vector_temporary, vectors)
            vector_temporary.replace(self._vector_file())
            temporary.replace(self.path)
        except OSError:
            self._dirty = True              # try again next time; the memory in hand is unaffected

    # ---- remembering ---------------------------------------------------------------------------
    def embed(self, crops):
        """Vectors for a list of HxWx3 uint8 crops: (n, dim) float32, each of length 1."""
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        return np.asarray(self.embedder.images(list(crops)), dtype=np.float32)

    def add(self, vector, x, y, label="", color=None, size_m=0., now=None, pixels=0):
        """Keep one sighting. If the same thing is already remembered at that spot (the same label there; or,
        without labels, something that looks the same), that entry is brought up to date instead: its position
        is the newer one, and its picture too unless the one already kept is a much better view (`pixels`: the
        size of the crop the vector was made from). Returns the entry."""
        now = self.clock() if now is None else now
        vector = np.asarray(vector, dtype=np.float32)
        with self.lock:
            best = None
            if self.entries:
                near = [i for i, e in enumerate(self.entries)
                        if np.hypot(e["x"] - x, e["y"] - y) <= SAME_SPOT_M + max(e.get("size_m") or 0., size_m or 0.) / 2]
                if near:
                    alike = self.vectors[near].astype(np.float32) @ vector
                    # The views already kept of this same thing, most like the new picture first.
                    views = [(float(alike[i]), near[i]) for i in np.argsort(-alike)
                             if ((self.entries[near[i]].get("label") or "") == label
                                 if (label and self.entries[near[i]].get("label")) else alike[i] >= SAME_LOOK)]
                    if views and (views[0][0] >= SAME_VIEW or not label):
                        best = views[0][1]                  # the same view again
                    elif len(views) >= MAX_VIEWS:           # a new view, and no room for one more: the oldest goes
                        best = min(views, key=lambda view: self.entries[view[1]].get("viewed", 0.))[1]
                        self.entries[best]["pixels"] = 0
            if best is not None:
                entry = self.entries[best]
                entry.update(t=now, x=round(float(x), 3), y=round(float(y), 3), seen=entry.get("seen", 1) + 1)
                if label:
                    entry["label"], entry["color"] = label, color
                if size_m:
                    entry["size_m"] = round(float(size_m), 2)
                if pixels >= KEEP_VIEW * entry.get("pixels", 0) or now - entry.get("viewed", 0.) > STALE_VIEW_S:
                    self.vectors[best] = vector
                    entry["pixels"], entry["viewed"] = int(pixels), now
            else:
                entry = {"id": f"v{self._next}", "t": now, "first": now, "x": round(float(x), 3), "y": round(float(y), 3),
                         "label": label or "", "color": color, "size_m": round(float(size_m or 0.), 2), "seen": 1,
                         "pixels": int(pixels), "viewed": now}
                self._next += 1
                self.entries.append(entry)
                self.vectors = np.concatenate([self.vectors, vector[None].astype(np.float16)])
                if len(self.entries) > MAX_ENTRIES:         # forget the sightings not refreshed for the longest
                    keep = sorted(np.argsort([e["t"] for e in self.entries])[-MAX_ENTRIES:])
                    self.entries = [self.entries[i] for i in keep]
                    self.vectors = self.vectors[keep]
            self._dirty = True
            return entry

    # ---- asking --------------------------------------------------------------------------------
    def text(self, phrase):
        """The vector for a phrase (kept for the next time it is asked)."""
        key = " ".join(str(phrase).lower().split())
        with self.lock:
            if key in self._texts:
                return self._texts[key]
        vector = np.asarray(self.embedder.texts([key]), dtype=np.float32)[0]
        with self.lock:
            self._texts[key] = vector
            while len(self._texts) > TEXT_CACHE:
                self._texts.pop(next(iter(self._texts)))
        return vector

    def scores(self, phrase, vectors):
        """How well each picture vector fits the phrase, 0 to 1 (the embedder's own decision scale: at or above
        `threshold` means "this is one")."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if not len(vectors):
            return np.zeros(0, np.float32)
        return np.asarray(self.embedder.match(self.text(phrase)[None], vectors), dtype=np.float32)[0]

    @property
    def threshold(self):
        return float(getattr(self.embedder, "threshold", .5))

    @property
    def strict_threshold(self):
        """The score to act on (walk up to something because it looks like the words), where a wrong answer
        costs more than a missed one; search, which only says where to look, uses the ordinary threshold."""
        return float(getattr(self.embedder, "strict_threshold", self.threshold))

    def search(self, phrase, limit=5, spread_m=SAME_SPOT_M):
        """The remembered things that fit the phrase: [(entry, score)], best first, one per place (several
        sightings within spread_m of each other are one thing: its best-fitting sighting stands for it)."""
        with self.lock:
            entries, vectors = list(self.entries), self.vectors.astype(np.float32)
        if not entries:
            return []
        scores = self.scores(phrase, vectors)
        found = []
        for i in np.argsort(-scores):
            if scores[i] < self.threshold or len(found) >= limit:
                break
            entry = entries[i]
            if any(np.hypot(entry["x"] - other["x"], entry["y"] - other["y"]) <= spread_m for other, _ in found):
                continue
            found.append((dict(entry), float(scores[i])))
        return found
