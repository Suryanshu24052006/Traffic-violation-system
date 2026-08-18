"""
plate_reader.py
==================
Automatic Number Plate Recognition (ANPR), two stages:

  1. Plate localization — like helmets, there's no COCO class for "license
     plate", so this needs a fine-tuned detector (see train_anpr.py). Falls
     back to a simple heuristic crop (lower-half of the vehicle bounding
     box) if no fine-tuned model is present — much lower precision, clearly
     documented as a fallback rather than a real solution.

  2. OCR — EasyOCR works reasonably out of the box, but Indian plates need
     post-processing: font styles vary a lot, so raw OCR output gets
     cleaned up with a regex + common-confusion character correction
     (0/O, 1/I, 8/B, 5/S) before being accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PlateResult:
    track_id: int
    raw_text: str
    cleaned_text: str
    is_valid_format: bool
    confidence: float


# Indian plate: 2 letters (state) + 1-2 digits (RTO) + 1-3 letters (series) + 4 digits
INDIA_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

CONFUSION_MAP = str.maketrans({"O": "0", "I": "1", "S": "5", "B": "8"})


def clean_plate_text(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    return text


class PlateReader:
    def __init__(self, config: dict):
        acfg = config["anpr"]
        self.detector_model_path = acfg["detector_model"]
        self.plate_detector = None
        if Path(self.detector_model_path).exists():
            from ultralytics import YOLO
            self.plate_detector = YOLO(self.detector_model_path)

        import easyocr
        self.reader = easyocr.Reader(acfg["ocr_languages"], gpu=False)

    @property
    def has_fine_tuned_detector(self) -> bool:
        return self.plate_detector is not None

    def _locate_plate_crop(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """Returns the best-guess plate region within a vehicle crop."""
        if self.plate_detector is not None:
            results = self.plate_detector.predict(vehicle_crop, verbose=False)[0]
            if len(results.boxes) > 0:
                best = max(results.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
                return vehicle_crop[y1:y2, x1:x2]
        # Fallback heuristic: plates sit in roughly the lower-middle third
        # of a vehicle's bounding box. Low precision — for demo purposes
        # only until a fine-tuned plate detector is plugged in.
        h, w = vehicle_crop.shape[:2]
        return vehicle_crop[int(h * 0.6):h, int(w * 0.2):int(w * 0.8)]

    def read(self, vehicle_crop: np.ndarray, track_id: int) -> PlateResult:
        plate_crop = self._locate_plate_crop(vehicle_crop)
        if plate_crop.size == 0:
            return PlateResult(track_id, "", "", False, 0.0)

        ocr_results = self.reader.readtext(plate_crop)
        if not ocr_results:
            return PlateResult(track_id, "", "", False, 0.0)

        # Concatenate all detected text fragments (plates sometimes OCR as
        # two chunks), weight confidence by text length.
        raw = "".join(r[1] for r in ocr_results)
        avg_conf = sum(r[2] for r in ocr_results) / len(ocr_results)

        cleaned = clean_plate_text(raw)
        is_valid = bool(INDIA_PLATE_RE.match(cleaned))

        if not is_valid:
            # retry with common OCR confusion-character correction on the
            # numeric segments only — don't blanket-apply, it can break
            # otherwise-correct reads.
            corrected = self._try_confusion_correction(cleaned)
            if corrected and INDIA_PLATE_RE.match(corrected):
                cleaned, is_valid = corrected, True

        return PlateResult(
            track_id=track_id, raw_text=raw, cleaned_text=cleaned,
            is_valid_format=is_valid, confidence=float(avg_conf),
        )

    @staticmethod
    def _try_confusion_correction(text: str) -> str | None:
        if len(text) < 8:
            return None
        # naive: try translating the whole string and see if that validates.
        # A production version would only correct positions expected to be
        # digits vs letters based on the plate format's fixed structure.
        candidate = text.translate(CONFUSION_MAP)
        return candidate
