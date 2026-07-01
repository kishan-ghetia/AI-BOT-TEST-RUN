"""Fair value gaps (imbalances): 3-candle gaps between bar1 and bar3 extremes."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FairValueGap:
    index: int  # middle candle index
    kind: str  # "bullish" | "bearish"
    top: float
    bottom: float


def find_fvgs(df: pd.DataFrame, min_gap_pct: float = 0.0) -> list[FairValueGap]:
    """Bullish FVG: low of bar i+1 > high of bar i-1 (gap left below).
    Bearish FVG: high of bar i+1 < low of bar i-1."""
    gaps: list[FairValueGap] = []
    high, low, close = df["high"], df["low"], df["close"]
    for i in range(1, len(df) - 1):
        ref = close.iloc[i]
        if ref <= 0:
            continue
        if low.iloc[i + 1] > high.iloc[i - 1]:
            size = (low.iloc[i + 1] - high.iloc[i - 1]) / ref
            if size >= min_gap_pct:
                gaps.append(FairValueGap(i, "bullish",
                                         top=low.iloc[i + 1],
                                         bottom=high.iloc[i - 1]))
        elif high.iloc[i + 1] < low.iloc[i - 1]:
            size = (low.iloc[i - 1] - high.iloc[i + 1]) / ref
            if size >= min_gap_pct:
                gaps.append(FairValueGap(i, "bearish",
                                         top=low.iloc[i - 1],
                                         bottom=high.iloc[i + 1]))
    return gaps


def unfilled_fvgs(df: pd.DataFrame, gaps: list[FairValueGap]) -> list[FairValueGap]:
    """Keep gaps price hasn't fully traded back through since formation."""
    out = []
    for g in gaps:
        after = df.iloc[g.index + 2 :]
        if g.kind == "bullish" and (after["low"] <= g.bottom).any():
            continue
        if g.kind == "bearish" and (after["high"] >= g.top).any():
            continue
        out.append(g)
    return out
