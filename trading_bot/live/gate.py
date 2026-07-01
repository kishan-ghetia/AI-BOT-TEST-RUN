"""Live-trading hard gate.

Live mode REFUSES to run unless, per (timeframe, symbol) archive:
  1. LIVE_TRADING_ENABLED=true is explicitly set in the environment/.env
  2. >= GATE_MIN_PAPER_WEEKS of *continuous* paper trading history
     (continuity is reset by Archive.update_performance if the bot was
     left off for more than 2x the timeframe cadence)
  3. Sharpe > GATE_MIN_SHARPE on the paper equity curve
  4. Max drawdown < GATE_MAX_DRAWDOWN

This is enforced in code: live/runner.py will not construct a real broker
until at least one requested (timeframe, symbol) passes this gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import config
from trading_bot.archive.store import Archive


@dataclass
class GateResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)


def check_live_eligibility(archive: Archive,
                           now: datetime | None = None) -> GateResult:
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []

    if not config.LIVE_TRADING_ENABLED:
        reasons.append("LIVE_TRADING_ENABLED is not set to true in .env")

    perf = archive.load_performance()
    started_at = perf.get("started_at")
    if started_at is None:
        reasons.append("no paper trading history at all")
    else:
        started = datetime.fromisoformat(started_at)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        min_age = timedelta(weeks=config.GATE_MIN_PAPER_WEEKS)
        age = now - started
        if age < min_age:
            reasons.append(
                f"paper run is {age.days}d old; needs "
                f">={config.GATE_MIN_PAPER_WEEKS:.0f} weeks continuous")

    sharpe = perf.get("sharpe")
    if sharpe is None:
        reasons.append("no Sharpe computed yet (too little history)")
    elif sharpe <= config.GATE_MIN_SHARPE:
        reasons.append(f"Sharpe {sharpe:.2f} <= required {config.GATE_MIN_SHARPE}")

    max_dd = perf.get("max_drawdown")
    if max_dd is None:
        reasons.append("no drawdown computed yet (too little history)")
    elif max_dd >= config.GATE_MAX_DRAWDOWN:
        reasons.append(
            f"max drawdown {max_dd:.1%} >= limit {config.GATE_MAX_DRAWDOWN:.0%}")

    return GateResult(allowed=not reasons, reasons=reasons)
