-- =============================================================================
-- Smart Traffic Violation & Risk Profiling System — DB Schema
-- =============================================================================
-- This is a MOCK / SIMULATED database standing in for a real state transport
-- department (RTO) + traffic police records system, PLUS a municipal
-- corporation accident-records system. Real government vehicle, challan,
-- DUI, hit-and-run, and accident black-spot databases are not publicly
-- accessible (and integrating with one would require formal government API
-- access / legal authorization). The schema below mirrors what such systems
-- would expose, so the pipeline and risk engine are written exactly as they
-- would be against real backends — only the data source is synthetic.

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------
-- Owners / drivers
-- -----------------------------------------------------------------------
-- IMPORTANT design note: a real system must identify a driver by a unique
-- ID (driving license number, or an Aadhaar-linked ID) — NOT by name.
-- Names collide ("Rahul Sharma" is not unique across even one city). This
-- is why `license_number` is the actual primary lookup key, and why
-- `get_owner_profile()` in db_manager.py takes a license number, not a
-- name string. A vehicle can only have one registered owner, but one
-- owner/driver can be linked to multiple vehicles — which is exactly what
-- makes a *driver*-level risk score different from a *vehicle*-level one
-- (e.g. a fleet operator with 5 trucks, each individually "low risk", who
-- is nonetheless accumulating violations across the fleet).
CREATE TABLE IF NOT EXISTS owners (
    owner_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    license_number TEXT NOT NULL UNIQUE,
    city           TEXT NOT NULL,
    state          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vehicles (
    plate_number      TEXT PRIMARY KEY,
    owner_id          INTEGER NOT NULL REFERENCES owners(owner_id),
    vehicle_type      TEXT NOT NULL CHECK (vehicle_type IN ('car','motorcycle','bus','truck')),
    registration_date TEXT NOT NULL,
    city              TEXT NOT NULL,
    state             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS challans (
    challan_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number  TEXT NOT NULL REFERENCES vehicles(plate_number),
    date          TEXT NOT NULL,
    violation_type TEXT NOT NULL,      -- e.g. 'no_helmet', 'overspeeding', 'truck_hours', 'signal_jump'
    fine_amount   INTEGER NOT NULL,
    paid_status   TEXT NOT NULL CHECK (paid_status IN ('paid','unpaid'))
);

CREATE TABLE IF NOT EXISTS intoxication_records (
    record_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number   TEXT NOT NULL REFERENCES vehicles(plate_number),
    date           TEXT NOT NULL,
    bac_level      REAL NOT NULL,       -- blood alcohol content, mg/100ml
    license_action TEXT NOT NULL        -- e.g. 'warning', 'suspended_30d', 'suspended_1y'
);

CREATE TABLE IF NOT EXISTS hit_and_run_cases (
    case_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number  TEXT NOT NULL REFERENCES vehicles(plate_number),
    date          TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('open','closed','under_investigation')),
    fir_number    TEXT NOT NULL
);

-- -----------------------------------------------------------------------
-- Cameras / locations + municipal accident history
-- -----------------------------------------------------------------------
-- A "camera" here means a fixed junction/road-segment install point — this
-- is what lets violations recorded over many separate video runs be
-- aggregated into a per-location pattern over time ("regularly
-- accident-prone", not just "this one clip had a lot of violations").
CREATE TABLE IF NOT EXISTS cameras (
    camera_id      TEXT PRIMARY KEY,
    location_name  TEXT NOT NULL,
    junction_type  TEXT,               -- 'signal', 'school_zone', 'highway_entry', 'flyover', ...
    city           TEXT NOT NULL,
    installed_date TEXT NOT NULL
);

-- Historical accident data for a location, as the municipal corporation /
-- traffic police would maintain it. Synthetic here (see note above) — a
-- real integration would pull this from a municipal road-safety database.
CREATE TABLE IF NOT EXISTS accident_black_spots (
    spot_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id               TEXT NOT NULL REFERENCES cameras(camera_id),
    historical_accident_count INTEGER NOT NULL DEFAULT 0,
    fatal_accident_count      INTEGER NOT NULL DEFAULT 0,
    last_incident_date        TEXT,
    official_risk_rating      TEXT CHECK (official_risk_rating IN ('low','medium','high','critical'))
);

-- -----------------------------------------------------------------------
-- Live pipeline output
-- -----------------------------------------------------------------------
-- Distinct from `challans` (pre-existing historical fines) — this is what
-- the CV pipeline itself WRITES TO as it processes footage. Includes
-- location fields so violations accumulate into a per-camera history that
-- the hotspot analyzer can mine for recurring patterns.
CREATE TABLE IF NOT EXISTS detected_violations (
    violation_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number     TEXT,               -- may be NULL if ANPR failed to read the plate
    track_id         INTEGER,
    camera_id        TEXT REFERENCES cameras(camera_id),
    timestamp        TEXT NOT NULL,
    violation_type   TEXT NOT NULL,      -- 'no_helmet' | 'overspeeding' | 'truck_hours'
    details          TEXT,               -- free-text, e.g. "82 km/h in 50 km/h zone"
    confidence        REAL,
    pixel_x           REAL,              -- within-frame location, for spatial hotspot analysis
    pixel_y           REAL,
    norm_x             REAL,              -- 0-1 normalized (comparable across resolutions)
    norm_y             REAL,
    snapshot_path      TEXT,
    risk_score_at_time REAL
);

-- -----------------------------------------------------------------------
-- Municipal alerts — audit trail of what was reported and when
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS municipal_alerts (
    alert_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id              TEXT NOT NULL REFERENCES cameras(camera_id),
    generated_at            TEXT NOT NULL,
    lookback_window_days    INTEGER,
    violation_count_window  INTEGER,
    dominant_violation_type TEXT,
    safety_priority_score   REAL,
    is_known_black_spot     INTEGER,     -- 0/1
    recommended_action      TEXT,
    report_path             TEXT
);

CREATE INDEX IF NOT EXISTS idx_challans_plate ON challans(plate_number);
CREATE INDEX IF NOT EXISTS idx_intox_plate ON intoxication_records(plate_number);
CREATE INDEX IF NOT EXISTS idx_har_plate ON hit_and_run_cases(plate_number);
CREATE INDEX IF NOT EXISTS idx_vehicles_owner ON vehicles(owner_id);
CREATE INDEX IF NOT EXISTS idx_detected_plate ON detected_violations(plate_number);
CREATE INDEX IF NOT EXISTS idx_detected_camera ON detected_violations(camera_id);
CREATE INDEX IF NOT EXISTS idx_black_spots_camera ON accident_black_spots(camera_id);
