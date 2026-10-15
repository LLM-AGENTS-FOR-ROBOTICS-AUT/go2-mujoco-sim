"""YOLO instance segmentation shared by the simulator (YOLO mode) and the real Go2.

Produces perception.Mask objects (COCO class names, full-resolution boolean masks), which
perception.fuse() turns into objects with LiDAR distance. ultralytics is AGPL-3.0 (see notices).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from .perception import Mask

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models/yolo11n-seg.pt"
WEIGHTS_SHA256 = {"yolo11n-seg.pt": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"}


class DetectorError(RuntimeError):
    pass


class YoloSegDetector:
    def __init__(self, weights=None, confidence=.25, device=None, imgsz=1280):
        self.imgsz = imgsz   # full camera resolution: small objects on the floor need it
        self.weights = Path(weights or os.environ.get("GO2_YOLO_WEIGHTS", DEFAULT_WEIGHTS))
        self.confidence = confidence
        self.device = device
        self.model = None
        self.lock = threading.Lock()

    def _load(self):
        if self.model is not None:
            return self.model
        if not self.weights.is_file():
            raise DetectorError(f"YOLO weights not found at {self.weights} (set GO2_YOLO_WEIGHTS).")
        expected = WEIGHTS_SHA256.get(self.weights.name)
        if expected:
            import hashlib
            if hashlib.sha256(self.weights.read_bytes()).hexdigest() != expected:
                raise DetectorError(f"{self.weights.name} checksum differs from the pinned file.")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise DetectorError("Object detection needs the ultralytics package (pip install ultralytics).") from None
        import torch
        if self.device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(str(self.weights))
        return self.model

    def __call__(self, rgb):
        """rgb: HxWx3 uint8 -> [Mask] at full image resolution."""
        import cv2
        with self.lock:
            model = self._load()
            result = model.predict(rgb[..., ::-1], conf=self.confidence, imgsz=self.imgsz, device=self.device,
                                   verbose=False)[0]
        h, w = rgb.shape[:2]
        masks = []
        if result.masks is None:
            return masks
        for i, polygons in enumerate(result.masks.xy):
            mask = np.zeros((h, w), dtype=np.uint8)
            if len(polygons):
                cv2.fillPoly(mask, [np.round(polygons).astype(np.int32)], 1)
            if mask.sum() < 12:
                continue
            label = result.names[int(result.boxes.cls[i])]
            masks.append(Mask(label, mask.astype(bool), float(result.boxes.conf[i])))
        return masks
