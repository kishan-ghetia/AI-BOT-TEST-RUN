"""Fibonacci retracement zones from swing points.

Exposed BOTH as a standalone Strategy and as a reusable filter
(`get_fib_zones` / `in_ote_zone`) that other strategies can consult.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.strategies import indicators as ta
from trading_bot.strategies.base import Signal, Strategy

# ICT "optimal trade entry" retracement band
OTE_LOW, OTE_HIGH = 0.62, 0.79


@dataclass
class FibZones:
    swing_high: float
    swing_low: float
    direction: str  # "up" leg (low->high) or "down" leg (high->low)

    def level(self, ratio: float) -> float:
        """Price at a retracement ratio of the most recent leg."""
        if self.direction == "up":
            return self.swing_high - ratio * (self.swing_high - self.swing_low)
        return self.swing_low + ratio * (self.swing_high - self.swing_low)

    def retracement_of(self, price: float) -> float:
        """How far price has retraced the leg (0=leg end, 1=leg start)."""
        rng = self.swing_high - self.swing_low
        if rng <= 0:
            return 0.0
        if self.direction == "up":
            return (self.swing_high - price) / rng
        return (price - self.swing_low) / rng


def get_fib_zones(df: pd.DataFrame, lookback: int = 3) -> FibZones | None:
    """Build fib zones from the two most recent alternating swing points."""
    sh_mask, sl_mask = ta.swing_points(df, lookback)
    highs = [(i, df["high"].iloc[i]) for i in range(len(df)) if sh_mask.iloc[i]]
    lows = [(i, df["low"].iloc[i]) for i in range(len(df)) if sl_mask.iloc[i]]
    if not highs or not lows:
        return None
    hi_idx, hi = highs[-1]
    lo_idx, lo = lows[-1]
    direction = "up" if lo_idx < hi_idx else "down"
    return FibZones(swing_high=hi, swing_low=lo, direction=direction)


def in_ote_zone(df: pd.DataFrame, lookback: int = 3) -> tuple[bool, str]:
    """Filter API: is current price inside the 62-79% OTE retracement zone?
    Returns (in_zone, trade_direction_suggested_by_leg)."""
    zones = get_fib_zones(df, lookback)
    if zones is None or len(df) == 0:
        return False, "none"
    retr = zones.retracement_of(df["close"].iloc[-1])
    side = "long" if zones.direction == "up" else "short"
    return bool(OTE_LOW <= retr <= OTE_HIGH), side


class FibonacciStrategy(Strategy):
    """Standalone signal: long when price retraces into OTE of an up leg,
    short when it retraces into OTE of a down leg."""

    name = "fibonacci"

    def __init__(self, lookback: int = 3):
        self.lookback = lookback

    def warmup_bars(self) -> int:
        return 30

    ANALYSIS_WINDOW = 150  # swings older than this aren't tradable zones

    def generate_signal(self, df: pd.DataFrame, context: dict) -> Signal:
        if len(df) < self.warmup_bars():
            return self.neutral(df, context, reason="warmup")
        df = df.tail(self.ANALYSIS_WINDOW)
        zones = get_fib_zones(df, self.lookback)
        if zones is None:
            return self.neutral(df, context, reason="no_swings")
        price = df["close"].iloc[-1]
        retr = zones.retracement_of(price)
        in_ote = OTE_LOW <= retr <= OTE_HIGH
        direction = 1.0 if zones.direction == "up" else -1.0
        if in_ote:
            value, conf = direction, 0.8
        elif 0.5 <= retr < OTE_LOW:
            value, conf = direction * 0.4, 0.4
        else:
            value, conf = 0.0, 0.1
        return Signal(
            symbol=context.get("symbol", "?"),
            timeframe=context.get("timeframe", "?"),
            timestamp=df.index[-1],
            strategy_name=self.name,
            value=value,
            confidence=conf,
            metadata={"retracement": float(retr), "in_ote": in_ote,
                      "leg": zones.direction},
        ).clamped()
