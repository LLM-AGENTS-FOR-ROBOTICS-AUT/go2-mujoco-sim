
"""Persistent storage for objects remembered by the Go2 robot."""

import json
import time
from pathlib import Path


class ObjectMemory:
    """Save and load remembered object positions using a JSON file."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.objects = []

        if self.path is not None:
            self.load()

    def load(self):
        """Load saved objects, or start empty if no save exists."""
        if self.path is None:
            self.objects = []
            return

        if not self.path.exists():
            self.objects = []
            return

        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, list):
                self.objects = [
                    item for item in data
                    if isinstance(item, dict)
                ]
            else:
                self.objects = []

        except (OSError, json.JSONDecodeError):
            self.objects = []

    def save(self):
        """Write remembered objects to disk."""
        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.objects, file, indent=2)

        temporary_path.replace(self.path)

    def remember(self, label, x, y, color=None, confidence=1.0, save=True):
        """Add or update an object's remembered position."""

        item = {
            "label": str(label),
            "x": float(x),
            "y": float(y),
            "color": color,
            "confidence": float(confidence),
            "last_seen": time.time(),
            "times_seen": 1,
        }

        for index, old in enumerate(self.objects):
            same_label = old.get("label") == item["label"]
            same_color = old.get("color") == item["color"]

            dx = old.get("x", 0.0) - item["x"]
            dy = old.get("y", 0.0) - item["y"]
            distance = (dx ** 2 + dy ** 2) ** 0.5

            if same_label and same_color and distance < 0.6:
                item["last_seen"] = time.time()
                item["times_seen"] = old.get("times_seen", 1) + 1
                self.objects[index] = item
                if save:
                    self.save()
                return

        self.objects.append(item)
        if save:
           self.save()

    def recall(self, label):
        """Return saved objects matching a label."""
        target = str(label).strip().lower()

        return [
            item for item in self.objects
            if item.get("label", "").strip().lower() == target
        ]
