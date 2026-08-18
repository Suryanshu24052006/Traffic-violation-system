"""
time_restriction.py
====================
Flags heavy vehicles (trucks/buses, configurable) present on a restricted
road outside their permitted entry window — mirroring real heavy-vehicle
entry-timing rules used in Indian metros (e.g. daytime truck bans on city
roads, allowed only late-night/early-morning).

The window is defined by a start/end time; if start > end it's treated as
an overnight window (e.g. 23:00 -> 07:00 means "allowed overnight, banned
during the day").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time


@dataclass
class TruckHoursResult:
    track_id: int
    class_name: str
    timestamp: datetime
    is_violation: bool
    reason: str


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


class TruckHoursChecker:
    def __init__(self, config: dict):
        tcfg = config["truck_hours"]
        self.restricted_classes = set(tcfg["restricted_classes"])
        self.window_start = _parse_hhmm(tcfg["allowed_window"]["start"])
        self.window_end = _parse_hhmm(tcfg["allowed_window"]["end"])

    def _within_allowed_window(self, t: time) -> bool:
        if self.window_start <= self.window_end:
            return self.window_start <= t <= self.window_end
        # overnight window, e.g. 23:00 -> 07:00
        return t >= self.window_start or t <= self.window_end

    def check(self, track_id: int, class_name: str,
              timestamp: datetime) -> TruckHoursResult | None:
        if class_name not in self.restricted_classes:
            return None

        allowed = self._within_allowed_window(timestamp.time())
        if allowed:
            return TruckHoursResult(
                track_id=track_id, class_name=class_name, timestamp=timestamp,
                is_violation=False, reason="Within permitted heavy-vehicle window.",
            )

        return TruckHoursResult(
            track_id=track_id, class_name=class_name, timestamp=timestamp,
            is_violation=True,
            reason=(
                f"{class_name} present at {timestamp.strftime('%H:%M')}, outside "
                f"permitted window "
                f"{self.window_start.strftime('%H:%M')}-{self.window_end.strftime('%H:%M')}."
            ),
        )
