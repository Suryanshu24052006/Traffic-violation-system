"""
hotspot_analyzer.py
=====================
Turns recurring violations at a camera location into a structured report a
municipal corporation / traffic police department could act on — this is
the "which junctions actually need attention" layer sitting on top of the
per-vehicle violation detection.

Two distinct outputs, deliberately kept separate:

  1. A MunicipalReport — aggregated, numeric, persisted to the DB
     (`municipal_alerts` table) and written as JSON + a human-readable
     Markdown summary. This is what you'd actually hand to a traffic
     engineer or forward to a municipal dashboard.

  2. A within-frame violation-density heatmap — WHERE inside the camera's
     view violations cluster (distinct from the general vehicle-presence
     heatmap in src/pipeline/heatmap.py, which shows traffic volume, not
     violations specifically).

Design note on "regularly accident-prone": a single video clip's worth of
violations isn't enough to call a location accident-prone — that requires
looking at accumulated history for that camera_id across many runs/days.
That's exactly why detected_violations persists camera_id + location to the
DB rather than staying in pipeline memory — this module queries that
accumulated history, not just the current run's in-memory state.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from database.db_manager import DBManager
from src.pipeline.heatmap import build_heatmap


@dataclass
class MunicipalReport:
    camera_id: str
    location_name: str
    generated_at: str
    lookback_days: int
    violation_count: int
    violation_breakdown: dict
    dominant_violation_type: str | None
    is_known_black_spot: bool
    historical_accident_count: int
    fatal_accident_count: int
    official_risk_rating: str | None
    safety_priority_score: float           # 0.0 - 1.0
    priority_level: str                    # low | medium | high | critical
    recommended_action: str
    explanation: list = field(default_factory=list)
    heatmap_path: str | None = None
    report_json_path: str | None = None
    report_md_path: str | None = None


class HotspotAnalyzer:
    def __init__(self, config: dict, db: DBManager):
        self.config = config
        self.db = db
        self.mcfg = config["municipal"]
        self.reports_dir = Path(self.mcfg.get("municipal_reports_dir",
                                               config["paths"]["municipal_reports_dir"]))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze_camera(self, camera_id: str, lookback_days: int | None = None) -> MunicipalReport:
        lookback_days = lookback_days or self.mcfg["lookback_days"]
        camera = self.db.get_camera(camera_id)
        if camera is None:
            raise ValueError(
                f"Unknown camera_id '{camera_id}' — register it via "
                f"DBManager.upsert_camera() before analyzing (see database/seed_data.py)."
            )

        since = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        violations = self.db.get_violations_for_camera(camera_id, since_iso=since)

        breakdown = Counter(v["violation_type"] for v in violations)
        dominant = breakdown.most_common(1)[0][0] if breakdown else None

        black_spot = self.db.get_black_spot(camera_id)
        is_known = black_spot is not None
        hist_accidents = black_spot["historical_accident_count"] if black_spot else 0
        fatal_accidents = black_spot["fatal_accident_count"] if black_spot else 0
        official_rating = black_spot["official_risk_rating"] if black_spot else None

        score, explanation = self._compute_safety_priority_score(
            violation_count=len(violations),
            historical_accident_count=hist_accidents,
            fatal_accident_count=fatal_accidents,
            official_rating=official_rating,
        )
        priority_level = self._categorize(score)
        recommended_action = self._recommend_action(dominant, priority_level)

        heatmap_path = self._build_violation_heatmap(camera_id, violations)

        report = MunicipalReport(
            camera_id=camera_id,
            location_name=camera["location_name"],
            generated_at=datetime.now().isoformat(),
            lookback_days=lookback_days,
            violation_count=len(violations),
            violation_breakdown=dict(breakdown),
            dominant_violation_type=dominant,
            is_known_black_spot=is_known,
            historical_accident_count=hist_accidents,
            fatal_accident_count=fatal_accidents,
            official_risk_rating=official_rating,
            safety_priority_score=score,
            priority_level=priority_level,
            recommended_action=recommended_action,
            explanation=explanation,
            heatmap_path=str(heatmap_path) if heatmap_path else None,
        )

        self._write_report(report)
        self.db.log_municipal_alert(
            camera_id=camera_id,
            generated_at=report.generated_at,
            lookback_window_days=lookback_days,
            violation_count_window=report.violation_count,
            dominant_violation_type=dominant,
            safety_priority_score=score,
            is_known_black_spot=is_known,
            recommended_action=recommended_action,
            report_path=report.report_json_path,
        )
        return report

    def analyze_all_cameras(self, lookback_days: int | None = None) -> list[MunicipalReport]:
        return [self.analyze_camera(cid, lookback_days) for cid in self.db.get_all_camera_ids()]

    # ------------------------------------------------------------------ #
    def _compute_safety_priority_score(self, violation_count, historical_accident_count,
                                        fatal_accident_count, official_rating) -> tuple[float, list]:
        w = self.mcfg["safety_priority_weights"]
        explanation = []

        density = min(violation_count / self.mcfg["violation_count_threshold_high"], 1.0)
        if violation_count > 0:
            explanation.append(
                f"{violation_count} violation(s) detected in the last "
                f"{self.mcfg['lookback_days']} days at this location."
            )

        hist_component = min(historical_accident_count / 20.0, 1.0)  # 20+ accidents on file = max
        if historical_accident_count > 0:
            explanation.append(f"{historical_accident_count} accident(s) on record historically.")

        fatal_component = min(fatal_accident_count / 5.0, 1.0)  # 5+ fatalities = max
        if fatal_accident_count > 0:
            explanation.append(f"{fatal_accident_count} of those were fatal — weighted heavily.")

        rating_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
        rating_component = rating_map.get(official_rating, 0.0)
        if official_rating:
            explanation.append(f"Official municipal risk rating on file: '{official_rating}'.")

        score = (
            w["violation_density"] * density
            + w["historical_accidents"] * hist_component
            + w["fatal_accidents"] * fatal_component
            + w["official_rating"] * rating_component
        )
        score = round(min(score, 1.0), 4)

        if not explanation:
            explanation.append("No violations or accident history on record for this window.")

        return score, explanation

    def _categorize(self, score: float) -> str:
        if score >= 0.75:
            return "critical"
        if score >= 0.50:
            return "high"
        if score >= 0.25:
            return "medium"
        return "low"

    def _recommend_action(self, dominant_violation_type: str | None, priority_level: str) -> str:
        actions = self.mcfg["recommended_actions"]
        base = actions.get(dominant_violation_type, actions["default"])
        if priority_level in ("high", "critical"):
            return f"PRIORITY ({priority_level.upper()}): {base}"
        return base

    def _build_violation_heatmap(self, camera_id: str, violations: list[dict]) -> Path | None:
        if not violations:
            return None
        w, h = self.mcfg["canonical_heatmap_size"]
        points = np.array([
            (v["norm_x"] * w, v["norm_y"] * h)
            for v in violations if v.get("norm_x") is not None and v.get("norm_y") is not None
        ])
        if points.size == 0:
            return None

        heatmap = build_heatmap(points, (h, w))
        import cv2
        out_path = self.reports_dir / f"{camera_id}_violation_heatmap.jpg"
        cv2.imwrite(str(out_path), heatmap)
        return out_path

    def _write_report(self, report: MunicipalReport):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.reports_dir / f"{report.camera_id}_{ts}.json"
        md_path = self.reports_dir / f"{report.camera_id}_{ts}.md"

        report.report_json_path = str(json_path)
        report.report_md_path = str(md_path)

        with open(json_path, "w") as f:
            json.dump(asdict(report), f, indent=2)

        with open(md_path, "w") as f:
            f.write(self._render_markdown(report))

    def _render_markdown(self, r: MunicipalReport) -> str:
        breakdown_lines = "\n".join(f"- {k}: {v}" for k, v in r.violation_breakdown.items()) or "- none"
        explanation_lines = "\n".join(f"- {line}" for line in r.explanation)
        black_spot_line = (
            f"**Known accident black spot** — {r.historical_accident_count} historical "
            f"accident(s), {r.fatal_accident_count} fatal, official rating: "
            f"{r.official_risk_rating}."
            if r.is_known_black_spot else
            "Not currently on the municipal black-spot register (absence of prior data "
            "isn't the same as 'safe' — just 'not yet flagged')."
        )

        return f"""# Municipal Traffic Safety Alert — {r.location_name} ({r.camera_id})

**Generated:** {r.generated_at}
**Window analyzed:** last {r.lookback_days} days
**Priority level:** {r.priority_level.upper()} (safety priority score: {r.safety_priority_score:.2f} / 1.00)

## Summary

{r.violation_count} violation(s) detected in this window. Dominant violation type:
**{r.dominant_violation_type or 'none'}**.

### Violation breakdown
{breakdown_lines}

### Accident history
{black_spot_line}

### Why this priority score
{explanation_lines}

## Recommended action

{r.recommended_action}

---
*This report is generated from a synthetic demo database — see README for why real
municipal integration would require formal data-sharing agreements. The scoring logic
and report structure are written as they would be for a real deployment.*
"""
