"""
Smoke tests for the municipal hotspot analyzer — isolated temp DB, no
video/model required.
    python -m pytest tests/test_hotspot.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db_manager import DBManager  # noqa: E402
from src.hotspot.hotspot_analyzer import HotspotAnalyzer  # noqa: E402


def load_config():
    with open(Path(__file__).parent.parent / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)
    # Redirect report output to a throwaway temp dir so test runs don't
    # pollute the real outputs/municipal_reports/ folder with test fixtures.
    config["municipal"]["municipal_reports_dir"] = tempfile.mkdtemp(prefix="municipal_reports_test_")
    return config


def fresh_db() -> DBManager:
    tmp_path = Path(tempfile.mkstemp(suffix=".db")[1])
    db = DBManager(str(tmp_path))
    db.init_schema()
    return db


def test_known_black_spot_with_many_violations_is_critical():
    db = fresh_db()
    db.upsert_camera("TCAM01", "Test Junction", "signal", "TestCity", "2020-01-01")
    db.add_black_spot("TCAM01", historical_accident_count=25, fatal_accident_count=6,
                       last_incident_date="2026-01-01", official_risk_rating="critical")

    for i in range(20):
        db.log_detected_violation(
            plate_number=f"T{i:04d}", track_id=i, camera_id="TCAM01",
            timestamp=datetime.now().isoformat(), violation_type="overspeeding",
            details="test", confidence=0.9, pixel_x=100, pixel_y=100,
            norm_x=0.3, norm_y=0.3, snapshot_path=None, risk_score_at_time=0.5,
        )

    analyzer = HotspotAnalyzer(load_config(), db)
    report = analyzer.analyze_camera("TCAM01")

    assert report.is_known_black_spot is True
    assert report.priority_level in ("high", "critical")
    assert "overspeeding" in report.recommended_action.lower() or \
           "speed" in report.recommended_action.lower()
    print(f"\n{report.priority_level} ({report.safety_priority_score}): {report.recommended_action}")


def test_unknown_location_with_no_violations_is_low_priority():
    db = fresh_db()
    db.upsert_camera("TCAM02", "Quiet Street", "signal", "TestCity", "2020-01-01")

    analyzer = HotspotAnalyzer(load_config(), db)
    report = analyzer.analyze_camera("TCAM02")

    assert report.is_known_black_spot is False
    assert report.priority_level == "low"
    assert report.violation_count == 0
    print(f"\n{report.priority_level} ({report.safety_priority_score})")


def test_unregistered_camera_raises_clear_error():
    db = fresh_db()
    analyzer = HotspotAnalyzer(load_config(), db)
    try:
        analyzer.analyze_camera("NOT_A_REAL_CAMERA")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "NOT_A_REAL_CAMERA" in str(e)
        print(f"\nCorrectly raised: {e}")


def test_report_persists_to_db_and_disk():
    db = fresh_db()
    db.upsert_camera("TCAM03", "Persist Test", "signal", "TestCity", "2020-01-01")
    analyzer = HotspotAnalyzer(load_config(), db)
    report = analyzer.analyze_camera("TCAM03")

    alerts = db.get_recent_municipal_alerts(5)
    assert len(alerts) == 1
    assert alerts[0]["camera_id"] == "TCAM03"
    assert Path(report.report_json_path).exists()
    assert Path(report.report_md_path).exists()
    print(f"\nPersisted report: {report.report_md_path}")


if __name__ == "__main__":
    test_known_black_spot_with_many_violations_is_critical()
    test_unknown_location_with_no_violations_is_low_priority()
    test_unregistered_camera_raises_clear_error()
    test_report_persists_to_db_and_disk()
    print("\nAll hotspot smoke tests passed.")
