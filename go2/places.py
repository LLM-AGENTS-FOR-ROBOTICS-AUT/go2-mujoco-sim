"""Named map-frame poses, persisted separately for each simulated scene.

The caller chooses a scene/frame-specific file. With no file, places live only
for the current connection; an odometry reset must not reuse old coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import unicodedata


def place_name(value):
    if not isinstance(value, str):
        raise ValueError("place name must be text")
    if any(unicodedata.category(c).startswith("C") for c in value):
        raise ValueError("place name cannot contain control characters")
    name = " ".join(unicodedata.normalize("NFKC", value).split())
    if not 1 <= len(name) <= 80:
        raise ValueError("place name must contain 1 to 80 characters")
    return name


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        value = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class Place:
    name: str
    x: float
    y: float
    yaw: float
    saved_at: float


class NamedPlaces:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._places = {}
        self._lock = threading.RLock()
        self.load_error = None
        self.load()

    def load(self):
        with self._lock:
            self._places.clear()
            self.load_error = None
            if self.path is None or not self.path.exists():
                return
            try:
                data = json.loads(self.path.read_text())
                if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("places"), list):
                    raise ValueError("unsupported named-place file")
                loaded = {}
                for row in data["places"]:
                    name = place_name(row["name"])
                    values = {k: _number(row[k], k) for k in ("x", "y", "yaw", "saved_at")}
                    key = name.casefold()
                    if key in loaded:
                        raise ValueError("duplicate place names")
                    loaded[key] = Place(name=name, **values)
                self._places = loaded
            except (OSError, ValueError, KeyError, TypeError) as error:
                # Refuse to navigate using partially recovered/malformed poses.
                self.load_error = f"Named places could not be loaded: {error}"

    def _save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "places": [asdict(p) for p in self.all()]}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=self.path.name + ".", suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                json.dump(payload, file, indent=2, allow_nan=False)
                file.write("\n")
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def remember(self, name, x, y, yaw, *, saved_at=None):
        name = place_name(name)
        x, y, yaw = (_number(v, k) for v, k in ((x, "x"), (y, "y"), (yaw, "yaw")))
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        stamp = _number(time.time() if saved_at is None else saved_at, "saved_at")
        with self._lock:
            key = name.casefold()
            previous = self._places.get(key)
            self._places[key] = Place(name, x, y, yaw, stamp)
            try:
                self._save()
            except OSError:
                if previous is None:
                    del self._places[key]
                else:
                    self._places[key] = previous
                raise
            self.load_error = None
            return self._places[key]

    def get(self, name):
        key = place_name(name).casefold()
        with self._lock:
            place = self._places.get(key)
        if place is None:
            raise ValueError(f"I do not know a place named {place_name(name)!r}; remember it first")
        return place

    def all(self):
        with self._lock:
            return sorted(self._places.values(), key=lambda p: p.name.casefold())

    def forget(self, name):
        with self._lock:
            place = self.get(name)
            del self._places[place.name.casefold()]
            try:
                self._save()
            except OSError:
                self._places[place.name.casefold()] = place
                raise
            return place
