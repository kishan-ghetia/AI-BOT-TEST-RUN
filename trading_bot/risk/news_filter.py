"""News risk filter: shrink or zero position sizing near high-impact
scheduled events (NFP, CPI, FOMC...). No-key => always 1.0 (no effect)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

HIGH_IMPACT = {"high", "3"}
MEDIUM_IMPACT = {"medium", "2"}

BLACKOUT_MINUTES = 30      # no new trades within +/- this window of high impact
CAUTION_MINUTES = 120      # reduced size within this window


def _event_time(event: dict) -> datetime | None:
    raw = event.get("time") or event.get("date")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def risk_multiplier(calendar_events: list[dict],
                    now: datetime | None = None) -> float:
    """1.0 = trade normally, 0.5 = half size, 0.0 = stand aside."""
    if not calendar_events:
        return 1.0
    now = now or datetime.now(timezone.utc)
    multiplier = 1.0
    for ev in calendar_events:
        impact = str(ev.get("impact", "")).lower()
        ts = _event_time(ev)
        if ts is None:
            continue
        delta = abs((ts - now).total_seconds()) / 60.0
        if impact in HIGH_IMPACT:
            if delta <= BLACKOUT_MINUTES:
                return 0.0
            if delta <= CAUTION_MINUTES:
                multiplier = min(multiplier, 0.5)
        elif impact in MEDIUM_IMPACT and delta <= BLACKOUT_MINUTES:
            multiplier = min(multiplier, 0.5)
    return multiplier
