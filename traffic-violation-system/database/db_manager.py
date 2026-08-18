"""
db_manager.py
=============
Thin data-access layer over the mock "fixed database" (SQLite standing in for
a real RTO / traffic-police / municipal-corporation records system — see
schema.sql for the rationale). Every other module (risk engine, hotspot
analyzer, pipeline, dashboard) should go through this class rather than
writing raw SQL inline.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class VehicleProfile:
    plate_number: str
    owner_name: Optional[str]
    owner_license_number: Optional[str]
    vehicle_type: Optional[str]
    city: Optional[str]
    state: Optional[str]
    found: bool
    challan_count: int = 0
    unpaid_challan_count: int = 0
    total_unpaid_fine: int = 0
    intoxication_flag: bool = False
    intoxication_count: int = 0
    hit_and_run_flag: bool = False
    open_hit_and_run_count: int = 0
    challan_history: list = field(default_factory=list)


@dataclass
class DriverProfile:
    """Aggregated across every vehicle registered to one license number —
    see schema.sql's note on why license_number (not name) is the key."""
    license_number: str
    name: Optional[str]
    found: bool
    vehicle_count: int = 0
    plate_numbers: list = field(default_factory=list)
    total_challan_count: int = 0
    total_unpaid_challan_count: int = 0
    total_unpaid_fine: int = 0
    intoxication_flag: bool = False
    intoxication_count: int = 0
    hit_and_run_flag: bool = False
    open_hit_and_run_count: int = 0


class DBManager:
    def __init__(self, db_path: str = "database/traffic_system.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA_PATH.read_text())

    # ------------------------------------------------------------------ #
    # Write path — seeding / historical records
    # ------------------------------------------------------------------ #
    def upsert_owner(self, name, license_number, city, state) -> int:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO owners (name, license_number, city, state)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(license_number) DO UPDATE SET
                       name=excluded.name, city=excluded.city, state=excluded.state""",
                (name, license_number, city, state),
            )
            row = conn.execute(
                "SELECT owner_id FROM owners WHERE license_number = ?", (license_number,)
            ).fetchone()
            return row["owner_id"]

    def upsert_vehicle(self, plate_number, owner_id, vehicle_type,
                        registration_date, city, state):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO vehicles (plate_number, owner_id, vehicle_type,
                       registration_date, city, state)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(plate_number) DO UPDATE SET
                       owner_id=excluded.owner_id,
                       vehicle_type=excluded.vehicle_type,
                       city=excluded.city,
                       state=excluded.state""",
                (plate_number, owner_id, vehicle_type, registration_date, city, state),
            )

    def add_challan(self, plate_number, date, violation_type, fine_amount, paid_status):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO challans (plate_number, date, violation_type,
                       fine_amount, paid_status) VALUES (?, ?, ?, ?, ?)""",
                (plate_number, date, violation_type, fine_amount, paid_status),
            )

    def add_intoxication_record(self, plate_number, date, bac_level, license_action):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO intoxication_records (plate_number, date, bac_level,
                       license_action) VALUES (?, ?, ?, ?)""",
                (plate_number, date, bac_level, license_action),
            )

    def add_hit_and_run_case(self, plate_number, date, status, fir_number):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO hit_and_run_cases (plate_number, date, status, fir_number)
                   VALUES (?, ?, ?, ?)""",
                (plate_number, date, status, fir_number),
            )

    def upsert_camera(self, camera_id, location_name, junction_type, city, installed_date):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO cameras (camera_id, location_name, junction_type, city, installed_date)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(camera_id) DO UPDATE SET
                       location_name=excluded.location_name,
                       junction_type=excluded.junction_type,
                       city=excluded.city""",
                (camera_id, location_name, junction_type, city, installed_date),
            )

    def add_black_spot(self, camera_id, historical_accident_count, fatal_accident_count,
                        last_incident_date, official_risk_rating):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO accident_black_spots
                       (camera_id, historical_accident_count, fatal_accident_count,
                        last_incident_date, official_risk_rating)
                   VALUES (?, ?, ?, ?, ?)""",
                (camera_id, historical_accident_count, fatal_accident_count,
                 last_incident_date, official_risk_rating),
            )

    # ------------------------------------------------------------------ #
    # Write path — live pipeline output
    # ------------------------------------------------------------------ #
    def log_detected_violation(self, plate_number, track_id, camera_id, timestamp,
                                violation_type, details, confidence,
                                pixel_x, pixel_y, norm_x, norm_y,
                                snapshot_path, risk_score_at_time):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO detected_violations
                       (plate_number, track_id, camera_id, timestamp, violation_type, details,
                        confidence, pixel_x, pixel_y, norm_x, norm_y, snapshot_path, risk_score_at_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (plate_number, track_id, camera_id, timestamp, violation_type, details,
                 confidence, pixel_x, pixel_y, norm_x, norm_y, snapshot_path, risk_score_at_time),
            )

    def log_municipal_alert(self, camera_id, generated_at, lookback_window_days,
                             violation_count_window, dominant_violation_type,
                             safety_priority_score, is_known_black_spot,
                             recommended_action, report_path):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO municipal_alerts
                       (camera_id, generated_at, lookback_window_days, violation_count_window,
                        dominant_violation_type, safety_priority_score, is_known_black_spot,
                        recommended_action, report_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (camera_id, generated_at, lookback_window_days, violation_count_window,
                 dominant_violation_type, safety_priority_score, int(is_known_black_spot),
                 recommended_action, report_path),
            )

    # ------------------------------------------------------------------ #
    # Read path — vehicle / driver profiles
    # ------------------------------------------------------------------ #
    def get_vehicle_profile(self, plate_number: str) -> VehicleProfile:
        with self._conn() as conn:
            vrow = conn.execute(
                """SELECT v.*, o.name AS owner_name, o.license_number AS owner_license_number
                   FROM vehicles v JOIN owners o ON v.owner_id = o.owner_id
                   WHERE v.plate_number = ?""",
                (plate_number,),
            ).fetchone()

            challans = conn.execute(
                "SELECT * FROM challans WHERE plate_number = ? ORDER BY date DESC",
                (plate_number,),
            ).fetchall()

            intox = conn.execute(
                "SELECT COUNT(*) AS c FROM intoxication_records WHERE plate_number = ?",
                (plate_number,),
            ).fetchone()

            har = conn.execute(
                """SELECT COUNT(*) AS c FROM hit_and_run_cases
                   WHERE plate_number = ? AND status != 'closed'""",
                (plate_number,),
            ).fetchone()

        if vrow is None:
            return VehicleProfile(
                plate_number=plate_number, owner_name=None, owner_license_number=None,
                vehicle_type=None, city=None, state=None, found=False,
            )

        unpaid = [c for c in challans if c["paid_status"] == "unpaid"]
        return VehicleProfile(
            plate_number=plate_number,
            owner_name=vrow["owner_name"],
            owner_license_number=vrow["owner_license_number"],
            vehicle_type=vrow["vehicle_type"],
            city=vrow["city"],
            state=vrow["state"],
            found=True,
            challan_count=len(challans),
            unpaid_challan_count=len(unpaid),
            total_unpaid_fine=sum(c["fine_amount"] for c in unpaid),
            intoxication_flag=intox["c"] > 0,
            intoxication_count=intox["c"],
            hit_and_run_flag=har["c"] > 0,
            open_hit_and_run_count=har["c"],
            challan_history=[dict(c) for c in challans],
        )

    def get_owner_profile(self, license_number: str) -> DriverProfile:
        """Aggregates violation/DUI/hit-and-run history across EVERY vehicle
        registered to this license number — this is what makes it a
        *driver*-level score rather than a single-vehicle score."""
        with self._conn() as conn:
            owner = conn.execute(
                "SELECT * FROM owners WHERE license_number = ?", (license_number,)
            ).fetchone()

            if owner is None:
                return DriverProfile(license_number=license_number, name=None, found=False)

            plates = conn.execute(
                "SELECT plate_number FROM vehicles WHERE owner_id = ?", (owner["owner_id"],)
            ).fetchall()
            plate_numbers = [p["plate_number"] for p in plates]

            if not plate_numbers:
                return DriverProfile(
                    license_number=license_number, name=owner["name"], found=True,
                    vehicle_count=0,
                )

            placeholders = ",".join("?" * len(plate_numbers))

            challans = conn.execute(
                f"SELECT * FROM challans WHERE plate_number IN ({placeholders})",
                plate_numbers,
            ).fetchall()
            unpaid = [c for c in challans if c["paid_status"] == "unpaid"]

            intox = conn.execute(
                f"""SELECT COUNT(*) AS c FROM intoxication_records
                    WHERE plate_number IN ({placeholders})""",
                plate_numbers,
            ).fetchone()

            har = conn.execute(
                f"""SELECT COUNT(*) AS c FROM hit_and_run_cases
                    WHERE plate_number IN ({placeholders}) AND status != 'closed'""",
                plate_numbers,
            ).fetchone()

        return DriverProfile(
            license_number=license_number,
            name=owner["name"],
            found=True,
            vehicle_count=len(plate_numbers),
            plate_numbers=plate_numbers,
            total_challan_count=len(challans),
            total_unpaid_challan_count=len(unpaid),
            total_unpaid_fine=sum(c["fine_amount"] for c in unpaid),
            intoxication_flag=intox["c"] > 0,
            intoxication_count=intox["c"],
            hit_and_run_flag=har["c"] > 0,
            open_hit_and_run_count=har["c"],
        )

    def get_all_plate_numbers(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT plate_number FROM vehicles").fetchall()
        return [r["plate_number"] for r in rows]

    def get_all_license_numbers(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT license_number FROM owners").fetchall()
        return [r["license_number"] for r in rows]

    def get_recent_detected_violations(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM detected_violations ORDER BY violation_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Read path — cameras / accident hotspots (used by hotspot_analyzer)
    # ------------------------------------------------------------------ #
    def get_all_camera_ids(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT camera_id FROM cameras").fetchall()
        return [r["camera_id"] for r in rows]

    def get_camera(self, camera_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_black_spot(self, camera_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM accident_black_spots WHERE camera_id = ? "
                "ORDER BY spot_id DESC LIMIT 1",
                (camera_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_black_spots(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT b.*, c.location_name, c.junction_type, c.city
                   FROM accident_black_spots b JOIN cameras c ON b.camera_id = c.camera_id
                   ORDER BY b.historical_accident_count DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def get_violations_for_camera(self, camera_id: str, since_iso: Optional[str] = None) -> list[dict]:
        with self._conn() as conn:
            if since_iso:
                rows = conn.execute(
                    """SELECT * FROM detected_violations
                       WHERE camera_id = ? AND timestamp >= ? ORDER BY timestamp""",
                    (camera_id, since_iso),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM detected_violations WHERE camera_id = ? ORDER BY timestamp",
                    (camera_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_municipal_alerts(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT a.*, c.location_name FROM municipal_alerts a
                   JOIN cameras c ON a.camera_id = c.camera_id
                   ORDER BY a.alert_id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
