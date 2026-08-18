# Model weights

| File | Status | Notes |
|---|---|---|
| `yolov8n.pt` | ✅ included (auto-downloaded on first run) | COCO-pretrained. Covers car/motorcycle/bus/truck detection out of the box — no fine-tuning required to get the base pipeline running. |
| `helmet_yolov8.pt` | ❌ not included — you train this | Fine-tune with `src/helmet/train_helmet.py`. See that file's docstring for dataset sources (Kaggle / Roboflow / IDD) and format. |
| `plate_yolov8.pt` | ❌ not included — you train this | Fine-tune with `src/anpr/train_anpr.py` for plate *localization*. OCR itself uses off-the-shelf EasyOCR. |

Until the helmet/plate weights exist, `HelmetDetector` and `PlateReader`
fall back to honest "unknown" / low-precision-heuristic behavior rather than
fabricating results — see their docstrings. This is intentional: a system
that can generate fines should never guess silently.
