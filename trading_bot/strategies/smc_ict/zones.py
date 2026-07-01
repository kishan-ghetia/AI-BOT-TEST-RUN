"""Premium/discount ranges and the ICT OTE zone (via the shared fib utility)."""
from __future__ import annotations

import pandas as pd

from trading_bot.strategies.fibonacci import get_fib_zones, in_ote_zone  # noqa: F401 (re-export)


def premium_discount(df: pd.DataFrame, lookback: int = 3) -> tuple[str, float]:
    """Where is price within the current dealing range?
    Returns (zone, position) with zone in {"premium","discount","equilibrium"}
    and position in [0,1] (0 = range low, 1 = range high)."""
    zones = get_fib_zones(df, lookback)
    if zones is None or len(df) == 0:
        return "equilibrium", 0.5
    rng = zones.swing_high - zones.swing_low
    if rng <= 0:
        return "equilibrium", 0.5
    pos = (df["close"].iloc[-1] - zones.swing_low) / rng
    pos = max(0.0, min(1.0, pos))
    if pos > 0.55:
        return "premium", pos
    if pos < 0.45:
        return "discount", pos
    return "equilibrium", pos
