"""
tracker.py
==========
Wraps `supervision`'s ByteTrack implementation to turn raw per-frame
detections into persistent tracks with stable IDs across frames — this is
what makes trajectories, speed estimation, and heatmaps possible at all.

Also maintains a rolling position history per track_id, which the speed
estimator and heatmap generator both consume.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import supervision as sv


class VehicleTracker:
    def __init__(self, config: dict, max_history: int = 60):
        tcfg = config["tracking"]
        self.tracker = sv.ByteTrack(
            track_activation_threshold=tcfg["track_activation_threshold"],
            lost_track_buffer=tcfg["lost_track_buffer"],
            minimum_matching_threshold=tcfg["minimum_matching_threshold"],
        )
        # Only track classes the system actually cares about — the raw YOLO
        # model also detects 'person' and everything else in COCO, and
        # without this filter those get tracked/heatmapped/annotated as if
        # they were vehicles too.
        self.allowed_classes = set(config["detection"]["vehicle_classes"])
        # track_id -> deque of (frame_idx, cx, cy)
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
        # track_id -> most recent class_name
        self.class_of: dict[int, str] = {}

    def update(self, yolo_results, frame_idx: int) -> sv.Detections:
        """
        yolo_results: raw ultralytics Results object for one frame
                      (from VehicleDetector.detect_raw).
        Returns a supervision.Detections object with .tracker_id populated,
        already filtered down to the configured vehicle classes.
        """
        detections = sv.Detections.from_ultralytics(yolo_results)

        if len(detections) > 0:
            names = yolo_results.names
            keep_mask = np.array([
                names.get(int(c)) in self.allowed_classes for c in detections.class_id
            ])
            detections = detections[keep_mask]

        detections = self.tracker.update_with_detections(detections)

        for i in range(len(detections)):
            tid = detections.tracker_id[i]
            if tid is None:
                continue
            x1, y1, x2, y2 = detections.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self.history[int(tid)].append((frame_idx, float(cx), float(cy)))

            cls_id = int(detections.class_id[i])
            cls_name = yolo_results.names.get(cls_id, "unknown")
            self.class_of[int(tid)] = cls_name

        return detections

    def get_trajectory(self, track_id: int) -> list[tuple[int, float, float]]:
        return list(self.history.get(track_id, []))

    def get_all_points(self) -> np.ndarray:
        """Flat array of every (x, y) point ever tracked — used for the heatmap."""
        pts = []
        for pts_deque in self.history.values():
            pts.extend([(p[1], p[2]) for p in pts_deque])
        return np.array(pts) if pts else np.empty((0, 2))
