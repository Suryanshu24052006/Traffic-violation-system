"""
train_anpr.py
===============
Fine-tunes a YOLOv8 detector for license-plate LOCALIZATION (the OCR step
itself uses off-the-shelf EasyOCR — see plate_reader.py — it's the
*detection* of where the plate is that benefits from fine-tuning).

------------------------------------------------------------------------
Data sources:
------------------------------------------------------------------------
  - Roboflow Universe "license plate detection" datasets (several
    India-specific ones exist, already in YOLO format).
  - Reference repos worth studying for approach (not for copy-pasting):
      sid0312/anpr_yolov5 — Indian-plate-specific YOLOv5 approach
      navaneet625/ModernLPR_system — YOLOv11 + regex correction, 97% char
        accuracy benchmark to aim for.

------------------------------------------------------------------------
Usage (same pattern as train_helmet.py):
------------------------------------------------------------------------
    python src/anpr/train_anpr.py --data path/to/plate_data.yaml --epochs 50

Then copy runs/anpr/finetune/weights/best.pt -> models/plate_yolov8.pt and
point anpr.detector_model at it in config.yaml.

------------------------------------------------------------------------
Fine-tuning notes specific to plates:
------------------------------------------------------------------------
  - Plates are SMALL objects relative to the frame — consider training at
    a higher imgsz (960-1280) than the vehicle detector, and/or tiling
    high-res frames, or detection recall on distant vehicles will be poor.
  - Class imbalance is less of an issue here than for helmets (every
    vehicle has exactly one plate), but motion blur at speed is the
    dominant failure mode — augment aggressively with blur/noise if your
    footage includes fast-moving vehicles.
  - After detection, OCR accuracy on Indian plates benefits from a second,
    smaller fine-tune: a font-specific OCR fine-tune (e.g. fine-tuning
    EasyOCR's recognition model, or training a small CRNN) on ~500-1000
    cropped plate images with your camera's actual resolution/angle. This
    is the "extreme fine-tuning" layer most tutorials skip — worth calling
    out explicitly in your README as the differentiator.
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = YOLO(args.base_model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device,
        project="runs/anpr", name="finetune", patience=15,
    )

    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f} | mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
