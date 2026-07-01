"""ICT session killzones (UTC): weight intraday signals by session timing."""
from __future__ import annotations

from datetime import datetime, time

# (start, end, weight) in UTC. Killzones get full weight, dead hours reduced.
KILLZONES = {
    "asian": (time(0, 0), time(4, 0), 0.6),
    "london_open": (time(7, 0), time(10, 0), 1.0),
    "ny_open": (time(12, 0), time(15, 0), 1.0),
    "london_close": (time(15, 0), time(17, 0), 0.8),
}
OFF_SESSION_WEIGHT = 0.4


def session_weight(ts: datetime | None, timeframe: str) -> tuple[float, str]:
    """Multiplier for signal confidence based on session. Daily bars are
    session-agnostic (weight 1)."""
    if timeframe == "daily" or ts is None:
        return 1.0, "all_day"
    t = ts.time() if hasattr(ts, "time") else None
    if t is None:
        return 1.0, "unknown"
    for name, (start, end, weight) in KILLZONES.items():
        if start <= t < end:
            return weight, name
    return OFF_SESSION_WEIGHT, "off_session"
