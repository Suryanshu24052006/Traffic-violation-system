"""
main_pipeline.py
===================
Ties every module together: detection -> tracking -> per-vehicle violation
checks (helmet / speed / truck-hours) -> ANPR -> DB lookup -> risk scoring
-> logging -> annotated output video + heatmap + summary report.

Usage:
    python -m src.pipeline.main_pipeline --video path/to/clip.mp4 \\
        --start-datetime "2026-08-18 14:30:00"

Design choices worth noting in a writeup:
  - Every stage degrades gracefully instead of crashing: an uncalibrated
    camera disables speed checks (not the whole pipeline); a helmet/plate
    model that hasn't been fine-tuned yet returns "unknown"/heuristic
    results instead of fabricating confident-looking output.
  - ANPR is only run when a violation is actually flagged (not on every
    vehicle every frame) — it's the most expensive step, and there's no
    reason to OCR a compliant vehicle's plate.
  - The risk score is recomputed and logged at the moment of each detected
    violation, using whatever DB history exists *at that time* — this
    mirrors how a real system would work (you don't get to retroactively
    know about a violation that hasn't happened yet).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.db_manager import DBManager  # noqa: E402
from src.detection.vehicle_detector import VehicleDetector  # noqa: E402
from src.tracking.tracker import VehicleTracker  # noqa: E402
from src.speed.speed_estimator import SpeedEstimator, UncalibratedCameraError  # noqa: E402
from src.truck_hours.time_restriction import TruckHoursChecker  # noqa: E402
from src.helmet.helmet_detector import HelmetDetector  # noqa: E402
from src.anpr.plate_reader import PlateReader  # noqa: E402
from src.risk.risk_engine import RiskEngine  # noqa: E402
from src.pipeline.heatmap import build_heatmap, overlay_heatmap  # noqa: E402
from src.hotspot.hotspot_analyzer import HotspotAnalyzer  # noqa: E402


ANPR_RECHECK_INTERVAL_FRAMES = 15  # avoid re-running OCR every single frame


class TrafficViolationPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db = DBManager(self.config["database"]["path"])
        self.db.init_schema()

        self.detector = VehicleDetector(self.config)
        self.tracker = VehicleTracker(self.config)
        self.truck_checker = TruckHoursChecker(self.config)
        self.helmet_detector = HelmetDetector(self.config)
        self.plate_reader = PlateReader(self.config)
        self.risk_engine = RiskEngine(self.config)
        self.hotspot_analyzer = HotspotAnalyzer(self.config, self.db)

        self.speed_estimator = None  # set once we know fps, see run()
        self.speed_disabled_reason = None

        # per-track_id bookkeeping across the run
        self.known_plate: dict[int, str] = {}
        self.session_violations: dict[int, set] = {}
        self.last_anpr_frame: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    def run(self, video_path: str, start_datetime: datetime,
             output_video_path: str | None = None, max_frames: int | None = None,
             camera_id: str | None = None, generate_municipal_report: bool = True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if camera_id and self.db.get_camera(camera_id) is None:
            print(f"[WARN] camera_id '{camera_id}' not registered — auto-registering as an "
                  f"unmapped location. Register it properly via DBManager.upsert_camera() "
                  f"for real municipal reporting (see database/seed_data.py).")
            self.db.upsert_camera(camera_id, f"Unregistered location ({camera_id})",
                                   "unknown", "unknown", datetime.now().strftime("%Y-%m-%d"))

        try:
            self.speed_estimator = SpeedEstimator(self.config, fps=fps)
        except UncalibratedCameraError as e:
            self.speed_disabled_reason = str(e)
            print(f"[WARN] Speed estimation disabled: {e}")

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        snapshots_dir = Path(self.config["paths"]["violation_snapshots_dir"])
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        violation_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            timestamp = start_datetime + timedelta(seconds=frame_idx / fps)

            raw = self.detector.detect_raw(frame)
            tracked = self.tracker.update(raw, frame_idx)

            for i in range(len(tracked)):
                track_id = tracked.tracker_id[i]
                if track_id is None:
                    continue
                track_id = int(track_id)
                cls_id = int(tracked.class_id[i])
                class_name = raw.names.get(cls_id, "unknown")
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w), min(y2, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]

                violations_this_frame = self._check_violations(
                    track_id, class_name, crop, timestamp, frame_idx
                )

                if violations_this_frame:
                    violation_count += len(violations_this_frame)
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    self._handle_violations(
                        track_id, class_name, crop, timestamp, frame_idx,
                        violations_this_frame, snapshots_dir, camera_id,
                        pixel_x=cx, pixel_y=cy, norm_x=cx / w, norm_y=cy / h,
                    )

                self._draw_box(frame, (x1, y1, x2, y2), track_id, class_name,
                                self.session_violations.get(track_id, set()))

            if writer:
                writer.write(frame)
            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        heatmap_path = self._save_heatmap((h, w))

        print(f"\nProcessed {frame_idx} frames.")
        print(f"Total violations flagged: {violation_count}")
        print(f"Heatmap saved to: {heatmap_path}")
        if output_video_path:
            print(f"Annotated video saved to: {output_video_path}")

        municipal_report = None
        if camera_id and generate_municipal_report:
            municipal_report = self.hotspot_analyzer.analyze_camera(camera_id)
            print(
                f"\n[MUNICIPAL REPORT] {municipal_report.location_name} ({camera_id}) — "
                f"priority={municipal_report.priority_level} "
                f"(score={municipal_report.safety_priority_score}) | "
                f"{municipal_report.recommended_action}"
            )
            print(f"Report written to: {municipal_report.report_md_path}")

        return {
            "frames_processed": frame_idx,
            "violations": violation_count,
            "heatmap_path": str(heatmap_path),
            "municipal_report": municipal_report,
        }

    # ------------------------------------------------------------------ #
    def _check_violations(self, track_id, class_name, crop, timestamp, frame_idx) -> list[str]:
        found = []
        self.session_violations.setdefault(track_id, set())

        # Truck / heavy-vehicle time restriction
        truck_result = self.truck_checker.check(track_id, class_name, timestamp)
        if truck_result and truck_result.is_violation:
            found.append("truck_hours")

        # Helmet (two-wheelers only)
        if class_name == "motorcycle" and self.helmet_detector.is_fine_tuned:
            helmet_result = self.helmet_detector.detect(crop, track_id)
            if helmet_result.status == "no_helmet":
                found.append("no_helmet")

        # Speed (needs enough trajectory history + a calibrated camera)
        if self.speed_estimator is not None:
            traj = self.tracker.get_trajectory(track_id)
            speed_result = self.speed_estimator.estimate(track_id, class_name, traj)
            if speed_result and speed_result.is_violation:
                found.append("overspeeding")

        new_violations = [v for v in found if v not in self.session_violations[track_id]]
        self.session_violations[track_id].update(found)
        return new_violations

    def _handle_violations(self, track_id, class_name, crop, timestamp, frame_idx,
                            violations, snapshots_dir: Path, camera_id: str | None,
                            pixel_x: float, pixel_y: float, norm_x: float, norm_y: float):
        # Only pay the ANPR cost when there's actually a violation to attach a plate to.
        should_run_anpr = (
            track_id not in self.known_plate
            and frame_idx - self.last_anpr_frame.get(track_id, -999) >= ANPR_RECHECK_INTERVAL_FRAMES
        )
        plate_number = self.known_plate.get(track_id)
        if should_run_anpr:
            self.last_anpr_frame[track_id] = frame_idx
            plate_result = self.plate_reader.read(crop, track_id)
            if plate_result.is_valid_format:
                plate_number = plate_result.cleaned_text
                self.known_plate[track_id] = plate_number

        profile = self.db.get_vehicle_profile(plate_number) if plate_number else \
            self.db.get_vehicle_profile("UNREAD_PLATE")
        risk_result = self.risk_engine.compute_risk_score(
            profile, current_session_violation_types=list(self.session_violations[track_id])
        )

        snapshot_path = snapshots_dir / f"track{track_id}_frame{frame_idx}.jpg"
        cv2.imwrite(str(snapshot_path), crop)

        for v_type in violations:
            self.db.log_detected_violation(
                plate_number=plate_number,
                track_id=track_id,
                camera_id=camera_id,
                timestamp=timestamp.isoformat(),
                violation_type=v_type,
                details=f"class={class_name}, risk={risk_result.risk_category}",
                confidence=1.0,
                pixel_x=pixel_x, pixel_y=pixel_y, norm_x=norm_x, norm_y=norm_y,
                snapshot_path=str(snapshot_path),
                risk_score_at_time=risk_result.risk_score,
            )
            print(
                f"[VIOLATION] t={timestamp.strftime('%H:%M:%S')} track={track_id} "
                f"class={class_name} type={v_type} plate={plate_number or 'UNREAD'} "
                f"risk={risk_result.risk_category} ({risk_result.risk_score})"
            )

    # ------------------------------------------------------------------ #
    def _draw_box(self, frame, xyxy, track_id, class_name, violations):
        x1, y1, x2, y2 = xyxy
        color = (0, 0, 255) if violations else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{track_id} {class_name}"
        if violations:
            label += f" [{','.join(violations)}]"
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _save_heatmap(self, frame_shape) -> Path:
        points = self.tracker.get_all_points()
        heatmap = build_heatmap(points, frame_shape)
        out_path = Path(self.config["paths"]["outputs_dir"]) / "trajectory_heatmap.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), heatmap)
        return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-datetime", default=None,
                         help="'YYYY-MM-DD HH:MM:SS', defaults to now")
    parser.add_argument("--output-video", default="outputs/annotated_output.mp4")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--camera-id", default=None,
                         help="Fixed camera/location ID (e.g. CAM01). Enables persistent "
                              "violation-location logging and an end-of-run municipal "
                              "hotspot report. See database/seed_data.py for pre-registered IDs.")
    parser.add_argument("--no-municipal-report", action="store_true",
                         help="Skip generating the municipal report even if --camera-id is set.")
    args = parser.parse_args()

    start_dt = (
        datetime.strptime(args.start_datetime, "%Y-%m-%d %H:%M:%S")
        if args.start_datetime else datetime.now()
    )

    pipeline = TrafficViolationPipeline(config_path=args.config)
    pipeline.run(
        video_path=args.video,
        start_datetime=start_dt,
        output_video_path=args.output_video,
        max_frames=args.max_frames,
        camera_id=args.camera_id,
        generate_municipal_report=not args.no_municipal_report,
    )


if __name__ == "__main__":
    main()
