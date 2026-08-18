"""
risk_engine.py
==============
Computes a transparent, explainable "risk probability score" per vehicle.

Design note (important — read before you present this in an interview):
This is deliberately a WEIGHTED-RULE engine, not a trained ML classifier.
A trained model needs labeled ground-truth outcomes (e.g. "did this vehicle
actually re-offend within 6 months?") to learn from. Since the underlying
database here is synthetic, training a classifier on it would just be
learning noise and presenting it with false confidence — that's worse than
not training one at all.

The honest engineering path is:
  1. Ship this transparent, auditable rule engine now (every score is
     explainable: "this vehicle is HIGH risk because of 2 unpaid challans
     + 1 intoxication record").
  2. If/when real historical outcome data is available, replace
     `compute_risk_score()` internals with a calibrated logistic regression
     or gradient-boosted model trained on real outcomes, keeping the same
     input feature contract so nothing downstream has to change.

That's the kind of judgment call that's actually worth documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from database.db_manager import DriverProfile, VehicleProfile


@dataclass
class RiskResult:
    plate_number: str
    risk_score: float          # 0.0 - 1.0
    risk_category: str         # low | medium | high | critical
    explanation: list[str] = field(default_factory=list)


@dataclass
class DriverRiskResult:
    license_number: str
    risk_score: float
    risk_category: str
    explanation: list[str] = field(default_factory=list)


CHALLAN_COUNT_CAP = 10
CURRENT_VIOLATIONS_CAP = 3
DRIVER_CHALLAN_COUNT_CAP = 20   # aggregated across every vehicle a driver owns


class RiskEngine:
    def __init__(self, config: dict):
        rc = config["risk_engine"]
        self.weights = rc["weights"]
        self.thresholds = rc["category_thresholds"]
        self.driver_weights = rc["driver_weights"]
        self.driver_thresholds = rc["driver_category_thresholds"]

    def compute_risk_score(
        self,
        profile: VehicleProfile,
        current_session_violation_types: Optional[list[str]] = None,
    ) -> RiskResult:
        current_session_violation_types = current_session_violation_types or []
        explanation = []
        w = self.weights

        if not profile.found:
            explanation.append(
                "Plate not found in database (unregistered / misread) — "
                "scored on current-session evidence only."
            )

        # --- factor 1: historical challan volume -----------------------
        challan_component = min(profile.challan_count, CHALLAN_COUNT_CAP) / CHALLAN_COUNT_CAP
        if profile.challan_count > 0:
            explanation.append(f"{profile.challan_count} past challan(s) on record.")

        # --- factor 2: unpaid challan ratio ------------------------------
        unpaid_ratio = (
            profile.unpaid_challan_count / profile.challan_count
            if profile.challan_count > 0 else 0.0
        )
        if profile.unpaid_challan_count > 0:
            explanation.append(
                f"{profile.unpaid_challan_count} unpaid challan(s), "
                f"₹{profile.total_unpaid_fine} outstanding."
            )

        # --- factor 3: intoxication flag --------------------------------
        intox_component = 1.0 if profile.intoxication_flag else 0.0
        if profile.intoxication_flag:
            explanation.append(
                f"{profile.intoxication_count} prior intoxicated-driving record(s)."
            )

        # --- factor 4: hit-and-run flag -----------------------------------
        har_component = 1.0 if profile.hit_and_run_flag else 0.0
        if profile.hit_and_run_flag:
            explanation.append(
                f"{profile.open_hit_and_run_count} open/under-investigation "
                f"hit-and-run case(s)."
            )

        # --- factor 5: violations detected THIS session -------------------
        current_component = min(
            len(current_session_violation_types), CURRENT_VIOLATIONS_CAP
        ) / CURRENT_VIOLATIONS_CAP
        if current_session_violation_types:
            explanation.append(
                f"Detected now: {', '.join(current_session_violation_types)}."
            )

        score = (
            w["past_challan_count"] * challan_component
            + w["unpaid_challan_ratio"] * unpaid_ratio
            + w["intoxication_flag"] * intox_component
            + w["hit_and_run_flag"] * har_component
            + w["current_session_violations"] * current_component
        )
        score = round(min(score, 1.0), 4)

        category = self._categorize(score)
        if not explanation:
            explanation.append("Clean record — no prior violations or flags found.")

        return RiskResult(
            plate_number=profile.plate_number,
            risk_score=score,
            risk_category=category,
            explanation=explanation,
        )

    def _categorize(self, score: float) -> str:
        t = self.thresholds
        if score >= t["high"]:
            return "critical"
        if score >= t["medium"]:
            return "high"
        if score >= t["low"]:
            return "medium"
        return "low"

    # ------------------------------------------------------------------ #
    # Driver-level (aggregated across every vehicle a person owns)
    # ------------------------------------------------------------------ #
    def compute_driver_risk_score(self, profile: DriverProfile) -> DriverRiskResult:
        w = self.driver_weights
        explanation = []

        if not profile.found:
            return DriverRiskResult(
                license_number=profile.license_number, risk_score=0.0,
                risk_category="low",
                explanation=["License number not found in database (unregistered driver)."],
            )

        if profile.vehicle_count == 0:
            explanation.append("No vehicles currently registered to this license.")

        challan_component = min(profile.total_challan_count, DRIVER_CHALLAN_COUNT_CAP) / DRIVER_CHALLAN_COUNT_CAP
        if profile.total_challan_count > 0:
            explanation.append(
                f"{profile.total_challan_count} total challan(s) across "
                f"{profile.vehicle_count} registered vehicle(s) "
                f"({', '.join(profile.plate_numbers)})."
            )

        unpaid_ratio = (
            profile.total_unpaid_challan_count / profile.total_challan_count
            if profile.total_challan_count > 0 else 0.0
        )
        if profile.total_unpaid_challan_count > 0:
            explanation.append(
                f"{profile.total_unpaid_challan_count} unpaid across their fleet, "
                f"₹{profile.total_unpaid_fine} outstanding."
            )

        intox_component = 1.0 if profile.intoxication_flag else 0.0
        if profile.intoxication_flag:
            explanation.append(f"{profile.intoxication_count} intoxicated-driving record(s).")

        har_component = 1.0 if profile.hit_and_run_flag else 0.0
        if profile.hit_and_run_flag:
            explanation.append(
                f"{profile.open_hit_and_run_count} open/under-investigation hit-and-run case(s)."
            )

        score = (
            w["total_challan_count"] * challan_component
            + w["unpaid_challan_ratio"] * unpaid_ratio
            + w["intoxication_flag"] * intox_component
            + w["hit_and_run_flag"] * har_component
        )
        score = round(min(score, 1.0), 4)

        t = self.driver_thresholds
        if score >= t["high"]:
            category = "critical"
        elif score >= t["medium"]:
            category = "high"
        elif score >= t["low"]:
            category = "medium"
        else:
            category = "low"

        if not explanation:
            explanation.append("Clean record across all registered vehicles.")

        return DriverRiskResult(
            license_number=profile.license_number, risk_score=score,
            risk_category=category, explanation=explanation,
        )
