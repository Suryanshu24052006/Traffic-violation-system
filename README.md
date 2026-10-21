# Traffic Violation & Risk Profiling System

**Live demo:** https://trafficviolation-suryanshu.streamlit.app/ (free tier, may take a minute to wake up)

![dashboard](docs/demo_dashboard.png)

![detection on a busy junction: 20 vehicles found](docs/demo_detection.png)

Code is in [`traffic-violation-system/`](traffic-violation-system/). Run the commands below from inside that folder.

A computer vision project on traffic camera footage. It detects and tracks vehicles with YOLOv8, checks a few violations (overspeeding, trucks in restricted hours, no helmet), reads the number plate when there's a violation, looks up that vehicle and its owner in a records database, and gives a risk score. Violations at the same camera build up over time into a hotspot report for that junction.

I got the idea from IIIT Hyderabad's [VIOLA project](https://blogs.iiit.ac.in/viola/) with Telangana Police.

## Status

What works now, without training anything:

- vehicle detection (car, motorcycle, bus, truck) with COCO-pretrained YOLOv8, about 0.87 confidence on the sample bus image
- tracking with ByteTrack, so each vehicle keeps its ID across frames
- truck/bus restricted-hours check, including night windows that cross midnight
- speed from a perspective transform. I checked the maths on a fake trajectory (25 m in 1 s = 90 km/h). It's off by default because every camera needs its own calibration, and it raises an error instead of giving wrong speeds
- the database, the vehicle-level and driver-level risk scores, and the hotspot reports
- the full pipeline on a test video: annotated video, snapshots, trajectory heatmap, violations logged to the DB
- the Streamlit dashboard

What doesn't work yet:

| Part | Problem | Plan |
|---|---|---|
| Helmet detection | COCO has no helmet class, so it returns "unknown" for now | fine-tune with `src/helmet/train_helmet.py` |
| Plate localisation | COCO has no plate class, so it uses a rough crop | fine-tune with `src/anpr/train_anpr.py` |
| Speed | needs calibration per camera | set `source_polygon` in `config/config.yaml` |
| Indian roads | pretrained YOLO misses auto-rickshaws and struggles with dense traffic | fine-tune on IDD |

So there are no accuracy numbers for helmet or plates yet. Those need the fine-tuning first.

## The database is fake

Real RTO / challan / police records aren't public, so `database/seed_data.py` fills the schema with made-up Indian records (45 owners, 72 vehicles, 8 cameras, 4 of them known accident spots). Everything else is written as if it were a real backend, so only `database/db_manager.py` would change with real data.

Owners are keyed by licence number, not name. One person can own several vehicles, and names clash all the time, so the licence number is the reliable key.

## How it fits together

```
video -> YOLOv8 detection -> ByteTrack tracking
      -> helmet / speed / truck-hours checks
      -> if there's a violation: read plate -> look up vehicle + owner -> risk score
      -> log to detected_violations (with camera id and position in frame)
      -> over many runs at one camera: hotspot report for that junction
```

Plates are only read when there's a violation, not for every vehicle in every frame, which saves a lot of time.

**Risk score.** A weighted rule score from 0 to 1 using past challans, unpaid challans, drunk-driving records and hit-and-run cases. Weights are in `config/config.yaml`. I used rules instead of a trained model because the data is synthetic, so a model would only learn my own fake patterns. There are two scores: one per vehicle, and one per driver that adds up everything across all their vehicles. That catches someone whose vehicles each look fine on their own but whose combined record is bad.

**Hotspot report.** One clip seeing a violation doesn't make a dangerous junction. The analyzer looks at the last 30 days of violations at a camera, combines that with the accident history for that spot, and gives a priority score, a suggested action based on the most common violation type there, and a heatmap of where in the frame violations happen. It writes a JSON and Markdown report and saves an alert row in the DB.

## Running it

```bash
pip install -r requirements.txt
python scripts/init_db.py          # create and seed the fake DB

python -m src.pipeline.main_pipeline --video your_clip.mp4 --start-datetime "2026-08-18 14:30:00"

# add --camera-id CAM02 to also save violation locations and make a hotspot report
streamlit run dashboard/app.py
```

The dashboard's lookup tabs accept the sample plates and licence numbers that `init_db.py` prints.

## What I'd measure once the models are fine-tuned

- mAP for vehicles, and separately for helmet / no helmet
- ID switches on a clip I label by hand
- plate accuracy per character and per full plate, by lighting and distance
- the false positive rate on violations, since this could end up issuing fines. I'd also note which conditions cause them (night, blur, occlusion)

## Files

```
config/config.yaml    thresholds, paths, weights
database/             schema, DB access, fake data
src/detection/        YOLO detector
src/tracking/         ByteTrack + trajectories
src/speed/            speed estimation
src/truck_hours/      restricted-hours check
src/helmet/           helmet detector + training script
src/anpr/             plate reader + training script
src/risk/             vehicle and driver risk scores
src/hotspot/          hotspot reports
src/pipeline/         runs everything + heatmaps
dashboard/app.py      Streamlit app
tests/                tests for the DB, risk engine and hotspot analyzer
```

## Note

This is a learning project and the data is synthetic. Using it for real fines would need much better accuracy, a person checking every violation before a fine goes out, legal access to real records, and rules for how long footage and plate data are kept.
