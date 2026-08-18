"""
speed_estimator.py
===================
Estimates real-world speed (km/h) from a tracked vehicle's pixel trajectory
using a perspective transform (homography) from camera pixel-space to a
real-world top-down plane, calibrated in meters.

IMPORTANT — calibration is per-camera, not optional:
`source_polygon` in config.yaml must be four pixel coordinates (in the
actual camera view) that correspond to a real-world rectangle of known
width/height (`target_width_m` / `target_height_m`) — e.g. the corners of a
marked lane segment, or two lamp posts of known spacing. Shipping this with
placeholder zeros means speed numbers are NOT meaningful until you calibrate
against your own footage. This module will raise clearly if you try to use
it uncalibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class UncalibratedCameraError(RuntimeError):
    pass


@dataclass
class SpeedResult:
    track_id: int
    class_name: str
    speed_kmph: float
    speed_limit_kmph: float
    is_violation: bool
    margin_kmph: float


class SpeedEstimator:
    def __init__(self, config: dict, fps: float):
        scfg = config["speed_estimation"]
        source = np.array(scfg["source_polygon"], dtype=np.float32)

        if np.all(source == 0):
            raise UncalibratedCameraError(
                "speed_estimation.source_polygon in config.yaml is still the "
                "placeholder [0,0]x4. Calibrate it against your camera's real "
                "view before trusting speed numbers (see module docstring)."
            )

        w, h = scfg["target_width_m"], scfg["target_height_m"]
        target = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        self.homography, _ = cv2.findHomography(source, target)
        self.fps = scfg.get("fps_override") or fps
        self.limits = scfg["speed_limits_kmph"][scfg["active_road_type"]]
        self.tolerance = scfg["tolerance_kmph"]

    def _to_world(self, px: float, py: float) -> tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.homography)
        return float(world[0][0][0]), float(world[0][0][1])

    def estimate(self, track_id: int, class_name: str,
                 trajectory: list[tuple[int, float, float]]) -> SpeedResult | None:
        """trajectory: list of (frame_idx, px, py), oldest first."""
        if len(trajectory) < 2:
            return None

        f0, x0, y0 = trajectory[0]
        f1, x1, y1 = trajectory[-1]
        if f1 == f0:
            return None

        wx0, wy0 = self._to_world(x0, y0)
        wx1, wy1 = self._to_world(x1, y1)

        dist_m = float(np.hypot(wx1 - wx0, wy1 - wy0))
        dt_s = (f1 - f0) / self.fps
        if dt_s <= 0:
            return None

        speed_mps = dist_m / dt_s
        speed_kmph = speed_mps * 3.6

        limit = self.limits.get(class_name, self.limits.get("car"))
        margin = speed_kmph - limit
        is_violation = margin > self.tolerance

        return SpeedResult(
            track_id=track_id,
            class_name=class_name,
            speed_kmph=round(speed_kmph, 1),
            speed_limit_kmph=limit,
            is_violation=is_violation,
            margin_kmph=round(margin, 1),
        )
