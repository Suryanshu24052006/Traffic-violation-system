"""
train_helmet.py
==================
Fine-tunes a YOLOv8 classification/detection head for helmet / no_helmet
on two-wheeler riders. This is the "extreme fine-tuning" piece of the
project — the part that's actually hard and actually differentiates this
from a tutorial.

------------------------------------------------------------------------
STEP 1 — get data (pick one):
------------------------------------------------------------------------
  a) Kaggle "Helmet Detection" datasets (search "helmet detection yolo" on
     Kaggle — several are pre-formatted in YOLO layout).
  b) IDD (India Driving Dataset, IIIT-Hyderabad, https://idd.insaan.iiit.ac.in/)
     — crop rider regions from the fuller scene annotations.
  c) Roboflow Universe — search "helmet detection", many public YOLO-format
     datasets with train/val/test splits already done, exportable directly
     in Ultralytics format.

Whatever you use, you need a `data.yaml` with two classes:
    names: ['helmet', 'no_helmet']

------------------------------------------------------------------------
STEP 2 — organize into YOLO format:
------------------------------------------------------------------------
    helmet_dataset/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt     (YOLO-format: class cx cy w h, normalized)
        data.yaml

------------------------------------------------------------------------
STEP 3 — run this script:
------------------------------------------------------------------------
    python src/helmet/train_helmet.py --data path/to/data.yaml --epochs 50

This fine-tunes FROM a COCO-pretrained yolov8n checkpoint (transfer
learning), which converges much faster and on far less data than training
from scratch — a few hundred well-labeled images is a reasonable starting
point for a portfolio-grade result.

------------------------------------------------------------------------
STEP 4 — plug the result in:
------------------------------------------------------------------------
Point `helmet.model` in config/config.yaml at the resulting
`runs/detect/train/weights/best.pt`, e.g.:
    helmet:
      model: "models/helmet_yolov8.pt"
(copy best.pt there). HelmetDetector will pick it up automatically.

------------------------------------------------------------------------
Write up in your README: final mAP, precision/recall PER CLASS (no_helmet
recall matters far more than overall accuracy — a missed violation is a
worse failure mode than a false alarm here), and what conditions it still
struggles with (night footage, heavy occlusion, riders wearing caps that
resemble helmets from a distance, etc). That failure-analysis section is
what makes this read as real engineering rather than a tutorial rerun.
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--base-model", default="yolov8n.pt",
                         help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="'cpu', '0', '0,1', etc.")
    args = parser.parse_args()

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/helmet",
        name="finetune",
        patience=15,          # early stopping
        augment=True,
        # Augmentation matters a lot here: real rider footage has motion
        # blur, varied lighting, and heavy occlusion. Consider bumping
        # hsv_h/hsv_v, mosaic, and mixup beyond ultralytics defaults if
        # your validation curves plateau early.
    )

    metrics = model.val()
    print("Fine-tuning complete.")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print("Per-class precision/recall — check runs/helmet/finetune/ for full breakdown.")


if __name__ == "__main__":
    main()
