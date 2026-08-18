"""
vehicle_detector.py
====================
Vehicle detection wrapper around Ultralytics YOLO.

Out of the box this loads a COCO-pretrained checkpoint (yolov8n.pt), which
already includes car / motorcycle / bus / truck as native classes — so
detection + basic classification works immediately with ZERO training.

For production accuracy on Indian roads (auto-rickshaws, dense occlusion,
non-standard vehicle shapes) you'd fine-tune on IDD / DriveIndia and point
`model_path` at those weights instead — see models/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# COCO class indices for the classes we treat as "vehicle"
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    xyxy: tuple          # (x1, y1, x2, y2) in pixels
    confidence: float
    class_name: str


class VehicleDetector:
    def __init__(self, config: dict):
        dcfg = config["detection"]
        self.model = YOLO(dcfg["model"])
        self.conf_threshold = dcfg["confidence_threshold"]
        self.device = dcfg.get("device", "cpu")
        self.allowed_classes = set(dcfg["vehicle_classes"])

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = COCO_VEHICLE_CLASSES.get(cls_id)
            if cls_name is None or cls_name not in self.allowed_classes:
                continue
            xyxy = tuple(box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append(Detection(xyxy=xyxy, confidence=conf, class_name=cls_name))
        return detections

    def detect_raw(self, frame: np.ndarray):
        """Returns the raw ultralytics Results object — used when the caller
        wants a supervision.Detections conversion (see tracker.py)."""
        return self.model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )[0]
