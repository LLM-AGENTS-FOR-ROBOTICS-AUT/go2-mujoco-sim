"""Persistent object sightings and conservatively verified missing locations."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid


class ObjectMemory:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.objects = []
        self._lock = threading.RLock()
        self._absence = {}  # session-local (count, monotonic observation time)
        self.load_error = None
        self.save_error = None
        self.load()

    @staticmethod
    def _validated(row):
        if not isinstance(row, dict) or not isinstance(row.get("label"), str) or not row["label"].strip():
            raise ValueError("object label is missing")
        out = dict(row)
        for key, default in (("x", None), ("y", None), ("confidence", 1.), ("last_seen", 0.)):
            value = out.get(key, default)
            if isinstance(value, bool) or value is None:
                raise ValueError(f"invalid {key}")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"invalid {key}")
            out[key] = value
        if not 0 <= out["confidence"] <= 1:
            raise ValueError("confidence must be between zero and one")
        if out.get("color") is not None and not isinstance(out["color"], str):
            raise ValueError("invalid colour")
        count = out.get("times_seen", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("invalid seen count")
        out["times_seen"] = count
        out["id"] = out.get("id") if isinstance(out.get("id"), str) and out["id"] else uuid.uuid4().hex
        out["state"] = out.get("state", "active")
        if out["state"] not in ("active", "missing"):
            raise ValueError("invalid object state")
        if out["state"] == "missing":
            out["missing_since"] = float(out.get("missing_since", out["last_seen"]))
            if not math.isfinite(out["missing_since"]):
                raise ValueError("invalid missing timestamp")
        return out

    def load(self):
        with self._lock:
            self.objects = []
            self._absence.clear()
            self.load_error = None
            if self.path is None or not self.path.exists():
                return
            try:
                data = json.loads(self.path.read_text())
                if not isinstance(data, list):
                    raise ValueError("object memory must be a list")
                seen = set()
                for row in data:
                    try:
                        item = self._validated(row)
                        if item["id"] in seen:
                            raise ValueError("duplicate object ID")
                    except (ValueError, TypeError):
                        self.load_error = "Some invalid saved object records were ignored"
                        continue
                    self.objects.append(item)
                    seen.add(item["id"])
            except (OSError, ValueError) as error:
                self.load_error = f"Object memory could not be loaded: {error}"

    def snapshot(self, *, include_missing=True):
        with self._lock:
            return copy.deepcopy([r for r in self.objects if include_missing or r.get("state", "active") == "active"])

    def save(self):
        with self._lock:
            if self.path is None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                        prefix=self.path.name + ".", suffix=".tmp", delete=False) as file:
                    temporary = Path(file.name)
                    json.dump(self.objects, file, indent=2, allow_nan=False)
                    file.write("\n")
                os.replace(temporary, self.path)
                self.save_error = None
            except (OSError, ValueError) as error:
                self.save_error = f"Object memory was not saved: {error}"
                raise
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def remember(self, label, x, y, color=None, confidence=1.0, save=True):
        item = self._validated(dict(label=label, x=x, y=y, color=color, confidence=confidence,
                                    last_seen=time.time(), times_seen=1, state="active"))
        with self._lock:
            for index, old in enumerate(self.objects):
                same = old["label"].casefold() == item["label"].casefold() and old.get("color") == color
                if same and math.hypot(old["x"]-item["x"], old["y"]-item["y"]) < .6:
                    item["id"], item["times_seen"] = old["id"], old.get("times_seen", 1)+1
                    self.objects[index] = item
                    break
            else:
                # A far-away lookalike might be another object. Appearance
                # identity belongs to #32; do not silently merge it here.
                self.objects.append(item)
            self._absence.pop(item["id"], None)
            if save:
                self.save()
            return dict(item)

    def observe_empty(self, clear_ids, *, observed_at, seen_ids=(), save=True):
        """Retire a location after three distinct, consecutive clear views.

        The caller must establish camera visibility, no occlusion, and direct
        empty-floor evidence. An absent detection alone is never sufficient.
        Pending evidence is not persisted across connections/restarts.
        """
        clear_ids, seen_ids = set(clear_ids), set(seen_ids)
        stamp = float(observed_at)
        if not math.isfinite(stamp):
            raise ValueError("observation time must be finite")
        changed = []
        with self._lock:
            for item in self.objects:
                key = item["id"]
                if key in seen_ids or key not in clear_ids:
                    self._absence.pop(key, None)
                    continue
                if item.get("state", "active") == "missing":
                    continue
                count, previous = self._absence.get(key, (0, -math.inf))
                if stamp-previous < .15:
                    continue
                count += 1
                self._absence[key] = count, stamp
                if count >= 3:
                    item.update(state="missing", missing_since=time.time(),
                                missing_reason="last known location verified clear")
                    self._absence.pop(key, None)
                    changed.append(dict(item))
            if changed and save:
                self.save()
        return changed

    def reset_observation_evidence(self):
        with self._lock:
            self._absence.clear()

    def recall(self, label):
        target = str(label).strip().casefold()
        return [item for item in self.snapshot(include_missing=False)
                if item["label"].strip().casefold() == target]

    def forget(self, object_id):
        with self._lock:
            match = next((r for r in self.objects if r["id"] == object_id), None)
            if match is None:
                raise ValueError("unknown remembered object")
            original = list(self.objects)
            self.objects = [r for r in self.objects if r["id"] != object_id]
            try:
                self.save()
            except (OSError, ValueError):
                self.objects = original
                raise
            self._absence.pop(object_id, None)
            return dict(match)
