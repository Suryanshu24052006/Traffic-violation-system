# Smart Traffic Violation & Risk Profiling System

An end-to-end computer-vision system that detects and tracks vehicles from
traffic camera footage, flags helmet / overspeeding / restricted-hour
violations for cars, bikes, and trucks, reads license plates (ANPR), looks
up each vehicle's — and each *driver's*, aggregated across every vehicle
they own — history in a records database, computes a transparent risk
score, and rolls recurring violations up into municipal accident-hotspot
alerts for the traffic/road-safety authority. The same shape as real
systems deployed by Indian traffic police (see
[IIIT-Hyderabad's VIOLA project](https://blogs.iiit.ac.in/viola/) with
Telangana Police).

This repo is a **working, tested skeleton** — detection, tracking, speed
math, truck-hour logic, the database, and the risk engine all run and are
verified end to end (see "What's verified working" below). Helmet detection
and plate localization are wired up as pluggable components that need
fine-tuning on real data to reach production accuracy — that fine-tuning
step is the actual hard/interesting part of this project, not a missing
afterthought. See `src/helmet/train_helmet.py` and `src/anpr/train_anpr.py`.

## Why a "fixed database" instead of a real government system

Real RTO / traffic-police challan, DUI, hit-and-run, and municipal
accident-history records aren't publicly accessible (and integrating with
one for a portfolio project isn't legally sensible). `database/schema.sql`
mirrors what such systems would expose; `database/seed_data.py` populates
it with clearly-synthetic, fabricated Indian records (fictional names,
random plates, fabricated license numbers). The pipeline, risk engine, and
hotspot analyzer are written exactly as they'd be against real backends —
only the data source is mocked. Swapping in a real data source later means
changing `database/db_manager.py` only.

**Why the DB is keyed by license number, not name:** a vehicle has exactly
one registered owner (`vehicles.owner_id`), but one person can own several
vehicles — a driver-level risk score has to aggregate across all of them,
and matching people by name string is unreliable (name collisions are
common; a license/Aadhaar-linked ID is the actual unique key a real system
would use). See the `owners` table in `schema.sql` for the full rationale.

## Architecture

```
video/CCTV feed
      │
      ▼
┌─────────────────┐   COCO-pretrained YOLOv8 (car/motorcycle/bus/truck)
│ VehicleDetector  │   works out of the box, no training needed
└────────┬─────────┘
         ▼
┌─────────────────┐   ByteTrack — persistent IDs across frames
│ VehicleTracker   │   builds per-vehicle trajectory history
└────────┬─────────┘
         │
   ┌─────┼──────────────┬─────────────────┐
   ▼     ▼               ▼                 ▼
Helmet  Speed          Truck-hours       Heatmap
check   estimator      checker           (from trajectories)
(needs  (needs camera  (time-of-day
 fine-   calibration)   rule engine)
 tuning)
   │     │               │
   └─────┴───────┬───────┘
                 ▼
       any violation flagged?
                 │  yes
                 ▼
        ┌─────────────────┐
        │  ANPR (plate)    │  only runs when there's something to attach
        └────────┬─────────┘  a plate to — not every vehicle every frame
                 ▼
        ┌─────────────────┐
        │  DB lookup        │  challans / DUI / hit-and-run history —
        └────────┬─────────┘  vehicle-level AND, via owner_id, driver-level
                 ▼             (aggregated across every vehicle they own)
        ┌─────────────────┐
        │  Risk Engine       │  transparent weighted score, 0.0-1.0
        └────────┬─────────┘  (vehicle score + separate driver score)
                 ▼
   logged to detected_violations (with camera_id + pixel/normalized
   location) + shown on dashboard
                 │
                 ▼  accumulates across many runs/days at the same camera
        ┌───────────────────┐
        │  Hotspot Analyzer   │  cross-references accident_black_spots,
        └────────┬───────────┘  computes a safety-priority score
                 ▼
   Municipal report (JSON + Markdown) + violation-density heatmap +
   municipal_alerts audit-trail row — "which junctions need attention"
```

## What's verified working right now (no training required)

- Vehicle detection + classification (car/motorcycle/bus/truck) — tested on
  real imagery, ~0.87 confidence on a sample bus.
- Multi-object tracking with persistent IDs and trajectory history.
- Truck/bus restricted-hours violation logic, including overnight windows.
- Speed math (perspective-transform based) — verified against a known
  synthetic trajectory (25m in 1s = 90 km/h, correctly flagged over a 50
  km/h limit). **Disabled by default** until you calibrate
  `speed_estimation.source_polygon` in `config/config.yaml` against your own
  camera — it fails loudly rather than silently returning wrong speeds.
- The mock database (normalized schema — `owners` → `vehicles` → `challans`
  / `intoxication_records` / `hit_and_run_cases`, plus `cameras` /
  `accident_black_spots` — 45 synthetic owners, 72 vehicles, 8 camera
  locations, 4 with black-spot history) and every query path.
- The **vehicle-level** risk engine — verified to correctly separate a
  clean vehicle (score ≈0), a heavily-flagged vehicle (score ≈0.89,
  "critical"), and an unread plate (handled gracefully, not as an error).
- The **driver-level** risk engine — verified to correctly aggregate
  challans/DUI/hit-and-run across every vehicle registered to one license
  number (tested against a synthetic 4-vehicle owner and a 3-vehicle
  "fleet" case), and to handle an unregistered license number gracefully.
- ANPR pipeline mechanics (crop → OCR → regex-clean → validate) — OCR itself
  works via EasyOCR; the *plate-localization* fallback heuristic is
  low-precision until you fine-tune a plate detector (see below).
- The **hotspot analyzer / municipal reporting** — verified against three
  cases: a known black spot with heavy recent violations (correctly scores
  "critical" and recommends action matching the dominant violation type),
  a location with no violations or accident history (correctly scores
  "low"), and an unregistered camera_id (raises a clear error instead of
  silently reporting on nothing). Generates a JSON report, a human-readable
  Markdown report, a violation-density heatmap, and a persisted
  `municipal_alerts` DB row per run.
- The full pipeline end-to-end on a test video: detects, tracks, flags a
  truck-hours violation, logs it to the DB with camera location and a risk
  score, saves a snapshot, writes an annotated output video, generates a
  trajectory heatmap, and — when run with `--camera-id` — automatically
  produces a municipal hotspot report at the end.
- The Streamlit dashboard boots and serves all five tabs (run output,
  violation log, vehicle risk lookup, driver risk lookup, municipal hotspot
  alerts).

## What needs fine-tuning before this is production-accurate

| Component | Why it needs real data | Where to start |
|---|---|---|
| Helmet detection | No COCO class for "helmet" — currently returns `unknown` honestly rather than guessing | `src/helmet/train_helmet.py` |
| Plate localization | No COCO class for "license plate" — currently falls back to a crude heuristic crop | `src/anpr/train_anpr.py` |
| Speed calibration | Every camera's pixel→real-world mapping is different | Recalibrate `source_polygon` in `config.yaml` against your footage |
| Indian-road robustness | COCO-pretrained YOLO misses auto-rickshaws, struggles with occlusion-heavy Indian traffic scenes | Fine-tune the base detector on IDD / DriveIndia (see prior project research) |

## Setup

```bash
pip install -r requirements.txt
python scripts/init_db.py                     # creates + seeds the mock DB

# basic run (no municipal reporting):
python -m src.pipeline.main_pipeline \
    --video path/to/your_clip.mp4 \
    --start-datetime "2026-08-18 14:30:00"

# with a registered camera location -> also persists violation locations
# and auto-generates a municipal hotspot report at the end of the run:
python -m src.pipeline.main_pipeline \
    --video path/to/your_clip.mp4 \
    --start-datetime "2026-08-18 14:30:00" \
    --camera-id CAM02        # see database/seed_data.py for pre-registered IDs

streamlit run dashboard/app.py
```

Look up any of the sample plate/license numbers `scripts/init_db.py` prints
after seeding, in the dashboard's Vehicle Risk Lookup / Driver Risk Lookup
tabs, to see the risk scoring live.

## Honest evaluation plan (do this before calling it "done")

Don't just report overall accuracy — for a violation-detection system,
per-class recall on the *violation* class matters far more than aggregate
accuracy (a missed no-helmet detection is a worse failure than a false
alarm). Once you've fine-tuned the helmet/plate models on real data, report:

- Detection: mAP50-95 for vehicle classes, separately for helmet/no_helmet.
- Tracking: MOTA/MOTP if you have ground-truth trajectories, or at minimum
  ID-switch rate on a manually reviewed clip.
- ANPR: character-level and full-plate accuracy, broken out by lighting/
  distance conditions.
- **False-positive rate on violations** — explicitly, since this system
  could plausibly generate real fines. Document what conditions cause false
  positives (night footage, motion blur, occlusion, unusual vehicle shapes)
  and what you'd do about them before trusting this in the field.

That failure-analysis section, written honestly, is what separates this
from a tutorial rerun.

## Driver-level risk & municipal hotspot reporting — design notes

Two pieces were added on top of the base per-vehicle pipeline, both aimed at
"who/where needs attention," not just "what happened right now":

**Driver-level risk** (`RiskEngine.compute_driver_risk_score`,
`DBManager.get_owner_profile`) aggregates challan/DUI/hit-and-run history
across every vehicle registered to one license number. This surfaces a
pattern a single-vehicle view can't: an operator with several vehicles,
each individually unremarkable, whose *combined* record is a real problem
(the classic fleet-compliance case). It's a separate score from the
vehicle-level one deliberately — a clean truck driven by a high-risk owner,
and a risky vehicle owned by an otherwise-clean owner, are different
situations a real system should distinguish.

**Municipal hotspot reporting** (`src/hotspot/hotspot_analyzer.py`) exists
because "this camera saw a violation" and "this location is regularly
accident-prone" are different claims — the second needs accumulated history,
not one clip. That's why `detected_violations` persists `camera_id` and
normalized frame-location on every write: the analyzer queries that
accumulated history for a `camera_id` (default 30-day window, configurable),
cross-references it against `accident_black_spots` (the municipal accident
register — synthetic here, see the DB rationale above), and produces:

- a `safety_priority_score` (violation density this window + historical
  accident count + fatal-accident count, weighted, + any existing official
  risk rating),
- a recommended action matched to the *dominant* violation type at that
  location (speed enforcement infrastructure for overspeeding, patrol
  timing for helmet compliance, signage/timing review for truck-hours),
- a violation-density heatmap for that location specifically (distinct from
  the general vehicle-presence heatmap — this one shows where violations
  cluster, not where traffic volume is highest),
- a persisted `municipal_alerts` row (audit trail of what was reported and
  when) plus a JSON + Markdown report file, suitable for actually handing to
  a traffic engineer.

Run `python -m src.pipeline.main_pipeline --video <clip> --camera-id CAM02`
to see this generated automatically at the end of a run, or trigger it
on-demand per camera from the dashboard's Municipal Hotspot Alerts tab.

## Project layout

```
config/config.yaml          all thresholds, paths, model paths, municipal-report weights
database/                   schema.sql, db_manager.py, seed_data.py
src/detection/               YOLO vehicle detector (works out of the box)
src/tracking/                 ByteTrack wrapper + trajectory history
src/speed/                    perspective-transform speed estimator
src/truck_hours/               time-of-day restriction checker
src/helmet/                    helmet detector (pluggable) + train_helmet.py
src/anpr/                      plate reader (pluggable) + train_anpr.py
src/risk/                      vehicle- AND driver-level risk-scoring engine
src/hotspot/                    municipal accident-hotspot analyzer + reports
src/pipeline/                  ties everything together + heatmap generation
dashboard/app.py               Streamlit UI (5 tabs)
tests/                        smoke tests for DB, risk engine, hotspot analyzer
scripts/init_db.py             one-shot DB init + seed
```

## Legal/ethical note

This is a portfolio/learning project. The database is entirely synthetic.
A real deployment generating actual fines would need: much higher accuracy
bars than a portfolio project targets, a human-review step before any fine
is issued (false positives have real consequences), formal legal
authorization to access real vehicle/challan records, and a privacy/data-
retention policy for captured footage and plate data.
