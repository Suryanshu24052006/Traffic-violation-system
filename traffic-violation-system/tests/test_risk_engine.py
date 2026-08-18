"""
Quick smoke tests for the DB layer + risk engine — run without any video/model.
    python -m pytest tests/test_risk_engine.py -v

Each test gets its own throwaway SQLite file (tempfile) so runs are
isolated and repeatable — reusing the shared demo DB across test runs would
silently accumulate challans/records and make assertions drift over time.
"""
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db_manager import DBManager  # noqa: E402
from src.risk.risk_engine import RiskEngine  # noqa: E402


def load_config():
    with open(Path(__file__).parent.parent / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def fresh_db() -> DBManager:
    tmp_path = Path(tempfile.mkstemp(suffix=".db")[1])
    db = DBManager(str(tmp_path))
    db.init_schema()
    return db


def test_clean_vehicle_is_low_risk():
    db = fresh_db()
    engine = RiskEngine(load_config())

    owner_id = db.upsert_owner("Test Owner", "TEST-LIC-0001", "TestCity", "TestState")
    db.upsert_vehicle("TEST01AB0001", owner_id, "car", "2020-01-01", "TestCity", "TestState")
    profile = db.get_vehicle_profile("TEST01AB0001")
    result = engine.compute_risk_score(profile)

    assert result.risk_category in ("low", "medium")
    assert 0.0 <= result.risk_score <= 1.0
    print(f"\nClean vehicle -> {result.risk_category} ({result.risk_score}): {result.explanation}")


def test_risky_vehicle_is_flagged_higher():
    db = fresh_db()
    engine = RiskEngine(load_config())

    plate = "TEST02CD0002"
    owner_id = db.upsert_owner("Risky Owner", "TEST-LIC-0002", "TestCity", "TestState")
    db.upsert_vehicle(plate, owner_id, "truck", "2019-05-01", "TestCity", "TestState")
    for _ in range(6):
        db.add_challan(plate, "2025-01-01", "overspeeding", 1000, "unpaid")
    db.add_intoxication_record(plate, "2025-02-01", 95.0, "suspended_1y")
    db.add_hit_and_run_case(plate, "2025-03-01", "open", "FIR-1234/2025")

    profile = db.get_vehicle_profile(plate)
    result = engine.compute_risk_score(profile, current_session_violation_types=["overspeeding", "no_helmet"])

    print(f"\nRisky vehicle -> {result.risk_category} ({result.risk_score}): {result.explanation}")
    assert result.risk_category in ("high", "critical")
    assert result.risk_score > 0.5


def test_unknown_plate_handled_gracefully():
    db = fresh_db()
    engine = RiskEngine(load_config())
    profile = db.get_vehicle_profile("NOTAREALPLATE99")
    result = engine.compute_risk_score(profile, current_session_violation_types=["no_helmet"])
    assert profile.found is False
    assert result.risk_score >= 0.0
    print(f"\nUnknown plate -> {result.risk_category} ({result.risk_score}): {result.explanation}")


def test_driver_level_aggregates_across_vehicles():
    db = fresh_db()
    engine = RiskEngine(load_config())

    owner_id = db.upsert_owner("Fleet Owner", "TEST-LIC-0003", "TestCity", "TestState")
    plates = ["TEST03AA0001", "TEST03AA0002", "TEST03AA0003"]
    for i, plate in enumerate(plates):
        db.upsert_vehicle(plate, owner_id, "truck", "2019-01-01", "TestCity", "TestState")
        for _ in range(3):
            db.add_challan(plate, "2025-01-01", "truck_hours", 2000, "unpaid")

    profile = db.get_owner_profile("TEST-LIC-0003")
    assert profile.vehicle_count == 3
    assert profile.total_challan_count == 9  # 3 vehicles x 3 challans each

    result = engine.compute_driver_risk_score(profile)
    print(f"\nFleet owner -> {result.risk_category} ({result.risk_score}): {result.explanation}")
    assert result.risk_score > 0.0


def test_unknown_driver_handled_gracefully():
    db = fresh_db()
    engine = RiskEngine(load_config())
    profile = db.get_owner_profile("NOT-A-REAL-LICENSE")
    assert profile.found is False
    result = engine.compute_driver_risk_score(profile)
    assert result.risk_category == "low"
    print(f"\nUnknown driver -> {result.risk_category} ({result.risk_score}): {result.explanation}")


if __name__ == "__main__":
    test_clean_vehicle_is_low_risk()
    test_risky_vehicle_is_flagged_higher()
    test_unknown_plate_handled_gracefully()
    test_driver_level_aggregates_across_vehicles()
    test_unknown_driver_handled_gracefully()
    print("\nAll risk-engine smoke tests passed.")
