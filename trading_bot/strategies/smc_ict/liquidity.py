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
    high_np = df["high"].to_numpy(dtype=float)
    low_np = df["low"].to_numpy(dtype=float)
    highs = [(i, high_np[i]) for i in sh_mask.to_numpy().nonzero()[0]]
    lows = [(i, low_np[i]) for i in sl_mask.to_numpy().nonzero()[0]]
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
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    for pool in pools:
        start = max(pool.indices) + 1
        if pool.kind == "highs":
            hits = ((high[start:] > pool.level) & (close[start:] < pool.level)).nonzero()[0]
            if hits.size:
                sweeps.append(Sweep(start + int(hits[0]), "high_sweep", pool.level))
        else:
            hits = ((low[start:] < pool.level) & (close[start:] > pool.level)).nonzero()[0]
            if hits.size:
                sweeps.append(Sweep(start + int(hits[0]), "low_sweep", pool.level))
    return sweeps
