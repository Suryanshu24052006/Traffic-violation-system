"""
seed_data.py
============
Populates the mock "fixed database" with synthetic-but-realistic Indian
records: drivers (owners, keyed by license number — see schema.sql for why
that's the real key, not name), their vehicles, challan history,
intoxication records, hit-and-run cases, plus municipal camera locations
and accident black-spot history. Stands in for real RTO/traffic-police/
municipal backends. All names, plates, license numbers, and case numbers
below are fabricated for demo purposes.

Run: python database/seed_data.py
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db_manager import DBManager  # noqa: E402

random.seed(42)

STATE_CODES = {
    "MH": ("Maharashtra", ["Mumbai", "Pune", "Nagpur"]),
    "DL": ("Delhi", ["New Delhi", "Dwarka", "Rohini"]),
    "KA": ("Karnataka", ["Bengaluru", "Mysuru"]),
    "TS": ("Telangana", ["Hyderabad", "Warangal"]),
    "TN": ("Tamil Nadu", ["Chennai", "Coimbatore"]),
    "UP": ("Uttar Pradesh", ["Lucknow", "Noida"]),
    "GJ": ("Gujarat", ["Ahmedabad", "Surat"]),
    "WB": ("West Bengal", ["Kolkata", "Howrah"]),
}

FIRST_NAMES = [
    "Arjun", "Priya", "Rohit", "Ananya", "Vikram", "Sneha", "Karan", "Divya",
    "Rahul", "Neha", "Aditya", "Pooja", "Sanjay", "Kavya", "Manoj", "Isha",
    "Ravi", "Meera", "Suresh", "Anjali",
]
LAST_NAMES = [
    "Sharma", "Verma", "Reddy", "Nair", "Iyer", "Patel", "Gupta", "Rao",
    "Singh", "Kumar", "Das", "Chatterjee", "Menon", "Joshi", "Desai",
]

VEHICLE_TYPES = ["car", "motorcycle", "bus", "truck"]
VEHICLE_TYPE_WEIGHTS = [0.35, 0.45, 0.08, 0.12]  # motorcycles dominate Indian traffic

VIOLATION_TYPES = [
    ("no_helmet", (100, 500)),
    ("overspeeding", (500, 2000)),
    ("truck_hours", (1000, 5000)),
    ("signal_jump", (500, 1000)),
    ("no_seatbelt", (100, 500)),
    ("wrong_lane", (500, 1000)),
]

JUNCTION_TYPES = ["signal", "school_zone", "highway_entry", "flyover", "market_area"]

used_plates: set = set()
used_licenses: set = set()


def random_plate() -> str:
    while True:
        state = random.choice(list(STATE_CODES.keys()))
        rto = random.randint(1, 50)
        series = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=random.choice([1, 2])))
        digits = random.randint(1, 9999)
        plate = f"{state}{rto:02d}{series}{digits:04d}"
        if plate not in used_plates:
            used_plates.add(plate)
            return plate


def random_license_number(state_code: str) -> str:
    while True:
        rto = random.randint(1, 50)
        year = random.randint(2005, 2023)
        serial = random.randint(1, 9999999)
        lic = f"{state_code}{rto:02d}{year}{serial:07d}"
        if lic not in used_licenses:
            used_licenses.add(lic)
            return lic


def random_date(days_back_max=1000) -> str:
    d = datetime.now() - timedelta(days=random.randint(1, days_back_max))
    return d.strftime("%Y-%m-%d")


def seed_owners_and_vehicles(db: DBManager, n_owners: int = 45):
    """Most owners have exactly 1 vehicle; a minority (fleet-like) have 2-4 —
    this is what makes driver-level risk aggregation meaningfully different
    from vehicle-level in the demo data."""
    plates_by_owner = []

    for _ in range(n_owners):
        state_code = random.choice(list(STATE_CODES.keys()))
        state_name, cities = STATE_CODES[state_code]
        city = random.choice(cities)
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        license_number = random_license_number(state_code)

        owner_id = db.upsert_owner(name, license_number, city, state_name)

        n_vehicles = random.choices([1, 2, 3, 4], weights=[0.70, 0.18, 0.08, 0.04])[0]
        owner_plates = []
        for _ in range(n_vehicles):
            plate = random_plate()
            vtype = random.choices(VEHICLE_TYPES, weights=VEHICLE_TYPE_WEIGHTS, k=1)[0]
            reg_date = random_date(days_back_max=3650)
            db.upsert_vehicle(plate, owner_id, vtype, reg_date, city, state_name)
            owner_plates.append(plate)

            is_risky = random.random() < 0.15
            n_challans = random.randint(3, 9) if is_risky else random.randint(0, 3)
            for _ in range(n_challans):
                v_type, fine_range = random.choice(VIOLATION_TYPES)
                fine = random.randint(*fine_range)
                paid = random.choices(["paid", "unpaid"], weights=[0.6, 0.4])[0]
                db.add_challan(plate, random_date(), v_type, fine, paid)

            if random.random() < 0.06:
                bac = round(random.uniform(35, 120), 1)  # mg/100ml; India legal limit is 30
                action = "suspended_1y" if bac > 80 else "suspended_30d"
                db.add_intoxication_record(plate, random_date(), bac, action)

            if random.random() < 0.03:
                status = random.choice(["open", "under_investigation", "closed"])
                fir = f"FIR-{random.randint(1000,9999)}/{random.choice(list(range(2022,2027)))}"
                db.add_hit_and_run_case(plate, random_date(), status, fir)

        plates_by_owner.append((license_number, owner_plates))

    return plates_by_owner


def seed_cameras_and_black_spots(db: DBManager):
    """A handful of fixed junction cameras across cities, a subset of which
    have real-looking historical accident data (i.e. known black spots)."""
    locations = [
        ("CAM01", "MG Road Signal Junction", "signal", "Bengaluru"),
        ("CAM02", "Silk Board Flyover", "flyover", "Bengaluru"),
        ("CAM03", "Cyber Towers Junction", "signal", "Hyderabad"),
        ("CAM04", "St. Mary's School Zone", "school_zone", "Hyderabad"),
        ("CAM05", "NH48 Highway Entry", "highway_entry", "Mumbai"),
        ("CAM06", "Dadar Market Area", "market_area", "Mumbai"),
        ("CAM07", "Connaught Place Circle", "signal", "New Delhi"),
        ("CAM08", "Anna Salai Junction", "signal", "Chennai"),
    ]

    for camera_id, location_name, junction_type, city in locations:
        db.upsert_camera(camera_id, location_name, junction_type, city,
                          random_date(days_back_max=1800))

    # A subset are known black spots with real accident history — the rest
    # have none on file (which itself is meaningful: no prior data doesn't
    # mean "safe", just "not yet flagged" — worth noting in the report).
    black_spot_config = [
        ("CAM01", 14, 2, "high"),
        ("CAM02", 27, 6, "critical"),
        ("CAM04", 8, 1, "medium"),
        ("CAM05", 19, 5, "high"),
    ]
    for camera_id, accidents, fatal, rating in black_spot_config:
        db.add_black_spot(
            camera_id, accidents, fatal, random_date(days_back_max=400), rating
        )


def seed(n_owners: int = 45):
    db = DBManager()
    db.init_schema()

    plates_by_owner = seed_owners_and_vehicles(db, n_owners)
    seed_cameras_and_black_spots(db)

    total_vehicles = sum(len(p) for _, p in plates_by_owner)
    print(f"Seeded {n_owners} owners / {total_vehicles} vehicles into {db.db_path}")
    print(f"Seeded 8 cameras, 4 with known accident black-spot history.")
    print(f"Sample license numbers you can query: {[p[0] for p in plates_by_owner[:5]]}")
    print(f"Sample plates you can query: {plates_by_owner[0][1]}")
    return plates_by_owner


if __name__ == "__main__":
    seed()
