"""
helmet_detector.py
====================
Detects helmet / no-helmet on riders of two-wheelers.

There is NO COCO class for "helmet" — unlike vehicle detection, this
genuinely requires a fine-tuned model. This module is written to be a drop-in
once you've trained one (see train_helmet.py), and falls back to a clearly
labeled "UNKNOWN" state if no fine-tuned weights are present, rather than
silently guessing — a system that flags fines should never fabricate a
result it has no evidence for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HelmetResult:
    track_id: int
    status: str          # "helmet" | "no_helmet" | "unknown"
    confidence: float


class HelmetDetector:
    def __init__(self, config: dict):
        hcfg = config["helmet"]
        self.conf_threshold = hcfg["confidence_threshold"]
        self.model_path = hcfg["model"]
        self.model = None

        if Path(self.model_path).exists():
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        # else: stays None -> every call returns "unknown", loudly, not silently.

    @property
    def is_fine_tuned(self) -> bool:
        return self.model is not None

    def detect(self, rider_crop: np.ndarray, track_id: int) -> HelmetResult:
        if self.model is None:
            return HelmetResult(track_id=track_id, status="unknown", confidence=0.0)

        results = self.model.predict(rider_crop, conf=self.conf_threshold, verbose=False)[0]
        if len(results.boxes) == 0:
            return HelmetResult(track_id=track_id, status="unknown", confidence=0.0)

        # Assumes a 2-class fine-tuned model: 0=helmet, 1=no_helmet
        # (adjust mapping to match your actual training config / data.yaml)
        best = max(results.boxes, key=lambda b: float(b.conf[0]))
        cls_id = int(best.cls[0])
        status = "helmet" if cls_id == 0 else "no_helmet"
        return HelmetResult(track_id=track_id, status=status, confidence=float(best.conf[0]))
