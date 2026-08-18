"""
heatmap.py
============
Builds a vehicle-density heatmap from every tracked position observed during
a run (VehicleTracker.get_all_points()). Manual accumulation + Gaussian blur
+ colormap — deliberately not dependent on any one CV library's built-in
heatmap annotator, so it doesn't break across supervision/opencv versions.
"""

from __future__ import annotations

import cv2
import numpy as np


def build_heatmap(points: np.ndarray, frame_shape: tuple[int, int],
                   blur_kernel: int = 51, alpha: float = 0.55) -> np.ndarray:
    """
    points: Nx2 array of (x, y) pixel coordinates.
    frame_shape: (height, width) of the frame the heatmap will be overlaid on.
    Returns an (H, W, 3) uint8 colormapped heatmap image (not yet overlaid).
    """
    h, w = frame_shape
    accum = np.zeros((h, w), dtype=np.float32)

    if points.size > 0:
        xs = np.clip(points[:, 0].astype(int), 0, w - 1)
        ys = np.clip(points[:, 1].astype(int), 0, h - 1)
        np.add.at(accum, (ys, xs), 1.0)

    if blur_kernel % 2 == 0:
        blur_kernel += 1
    accum = cv2.GaussianBlur(accum, (blur_kernel, blur_kernel), 0)

    if accum.max() > 0:
        norm = (accum / accum.max() * 255).astype(np.uint8)
    else:
        norm = accum.astype(np.uint8)

    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return colored


def overlay_heatmap(base_frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    return cv2.addWeighted(heatmap, alpha, base_frame, 1 - alpha, 0)
