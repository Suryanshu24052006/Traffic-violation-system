"""Convenience entrypoint: initialize schema + seed synthetic data in one step.
    python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.seed_data import seed  # noqa: E402

if __name__ == "__main__":
    seed()
