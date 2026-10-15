"""Download the Google Scanned Objects used in the room (textured meshes only, ~45 MB).

Source: github.com/kevinzakka/mujoco_scanned_objects at a pinned commit. The 3D assets are
CC-BY 4.0 (Google LLC, "Google Scanned Objects"); see THIRD_PARTY_NOTICES.md. The setup scripts run this
with --agent / -Agent:

  .venv/bin/python simulation/fetch_gso.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

COMMIT = "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c"
BASE = f"https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/{COMMIT}/models"
DEST = Path(__file__).resolve().parents[1] / "_deps/gso"
# model folder -> (COCO class YOLO knows it as, short name used in the room)
OBJECTS = {
    "Lovable_Huggable_Cuddly_Boutique_Teddy_Bear_Beige": ("teddy bear", "teddy_bear"),
    "Jansport_School_Backpack_Blue_Streak": ("backpack", "backpack"),
    "Top_Paw_Dog_Bowl_Blue_Paw_Bone_Ceramic_25_fl_oz_total": ("bowl", "dog_bowl"),
    "ACE_Coffee_Mug_Kristen_16_oz_cup": ("cup", "mug"),
    "Dog": ("dog", "toy_dog"),
    "Elephant": ("elephant", "toy_elephant"),
    "Breyer_Horse_Of_The_Year_2015": ("horse", "toy_horse"),
    "Crosley_Alarm_Clock_Vintage_Metal": ("clock", "alarm_clock"),
    "FIRE_TRUCK": ("truck", "fire_truck"),
    "Travel_Mate_P_series_Notebook": ("laptop", "laptop"),
    "Kanex_MultiSync_Wireless_Keyboard": ("keyboard", "keyboard"),
    "Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book": ("book", "book"),
}


def main():
    manifest = {}
    for folder, (label, short) in OBJECTS.items():
        target = DEST / short
        target.mkdir(parents=True, exist_ok=True)
        for name in ("model.obj", "texture.png"):
            path = target / name
            if not path.exists():
                with urllib.request.urlopen(f"{BASE}/{folder}/{name}", timeout=120) as response:
                    path.write_bytes(response.read())
            manifest[f"{short}/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{short:14s} {label:12s} {sum(p.stat().st_size for p in target.iterdir()) / 1e6:.1f} MB", flush=True)
    (DEST / "SOURCE.json").write_text(json.dumps({"repository": "kevinzakka/mujoco_scanned_objects", "commit": COMMIT,
                                                  "license": "CC-BY 4.0 (assets), MIT (MJCF)", "objects": OBJECTS,
                                                  "sha256": manifest}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
