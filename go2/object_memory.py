
"""Persistent storage for objects remembered by the Go2 robot."""

import json
from pathlib import Path


class ObjectMemory:
    """Save and load remembered object positions using a JSON file."""

    def __init__(self, path=None):
        if path is None:
            path = Path(__file__).resolve().parent / "object_memory.json"

        self.path = Path(path)
        self.objects = []

        self.load()

    def load(self):
        """Load saved objects, or start empty if no save exists."""
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
        self.path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.objects, file, indent=2)

        temporary_path.replace(self.path)

    def remember(self, label, x, y, color=None, confidence=1.0):
        """Add or update an object's remembered position."""
        item = {
            "label": str(label),
            "x": float(x),
            "y": float(y),
            "color": color,
            "confidence": float(confidence),
        }

        for index, old in enumerate(self.objects):
            if (
                old.get("label") == item["label"]
                and old.get("color") == item["color"]
            ):
                self.objects[index] = item
                self.save()
                return

        self.objects.append(item)
        self.save()

    def recall(self, label):
        """Return saved objects matching a label."""
        target = str(label).strip().lower()

        return [
            item for item in self.objects
            if item.get("label", "").strip().lower() == target
        ]
