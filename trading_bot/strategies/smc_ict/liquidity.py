"""Liquidity pools (equal highs/lows) and stop-hunt sweep detection."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.strategies import indicators as ta


@dataclass
class LiquidityPool:
    kind: str  # "highs" (buy-side liquidity above) | "lows" (sell-side below)
    level: float
    indices: list[int]


@dataclass
class Sweep:
    index: int
    kind: str  # "high_sweep" (stop hunt above then reject) | "low_sweep"
    level: float


def find_liquidity_pools(
    df: pd.DataFrame, tolerance_pct: float = 0.001, lookback: int = 3
) -> list[LiquidityPool]:
    """Two+ swing highs (or lows) within tolerance => resting liquidity."""
    sh_mask, sl_mask = ta.swing_points(df, lookback)
    highs = [(i, df["high"].iloc[i]) for i in range(len(df)) if sh_mask.iloc[i]]
    lows = [(i, df["low"].iloc[i]) for i in range(len(df)) if sl_mask.iloc[i]]
    pools: list[LiquidityPool] = []

    def cluster(points: list[tuple[int, float]], kind: str):
        used: set[int] = set()
        for a, (i, lvl) in enumerate(points):
            if i in used:
                continue
            group = [(i, lvl)]
            for j, lvl2 in points[a + 1 :]:
                if lvl > 0 and abs(lvl2 - lvl) / lvl <= tolerance_pct:
                    group.append((j, lvl2))
            if len(group) >= 2:
                used.update(g[0] for g in group)
                pools.append(LiquidityPool(
                    kind=kind,
                    level=sum(g[1] for g in group) / len(group),
                    indices=[g[0] for g in group],
                ))

    cluster(highs, "highs")
    cluster(lows, "lows")
    return pools


def find_sweeps(df: pd.DataFrame, pools: list[LiquidityPool]) -> list[Sweep]:
    """Sweep: a bar wicks through a pool level but closes back on the other
    side (stop hunt / liquidity grab), after the pool formed."""
    sweeps: list[Sweep] = []
    for pool in pools:
        start = max(pool.indices) + 1
        for i in range(start, len(df)):
            bar = df.iloc[i]
            if pool.kind == "highs" and bar["high"] > pool.level and bar["close"] < pool.level:
                sweeps.append(Sweep(i, "high_sweep", pool.level))
                break
            if pool.kind == "lows" and bar["low"] < pool.level and bar["close"] > pool.level:
                sweeps.append(Sweep(i, "low_sweep", pool.level))
                break
    return sweeps
