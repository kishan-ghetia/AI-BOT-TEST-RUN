"""Market structure: swing points, break of structure (BOS), change of character (CHoCH)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.strategies import indicators as ta


@dataclass
class StructureState:
    trend: str  # "bull" | "bear" | "range"
    last_bos_index: int | None
    last_choch_index: int | None
    swing_highs: list[tuple[int, float]]
    swing_lows: list[tuple[int, float]]


def analyze_structure(df: pd.DataFrame, lookback: int = 3) -> StructureState:
    """Walk bars chronologically tracking swings; a close beyond the latest
    swing extreme in trend direction = BOS, against it = CHoCH (trend flip)."""
    sh_mask, sl_mask = ta.swing_points(df, lookback)
    sh = sh_mask.to_numpy()
    sl = sl_mask.to_numpy()
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    trend = "range"
    last_bos = None
    last_choch = None

    for i in range(len(df)):
        if sh[i]:
            swing_highs.append((i, high[i]))
        if sl[i]:
            swing_lows.append((i, low[i]))
        if not swing_highs or not swing_lows:
            continue
        last_high = swing_highs[-1][1]
        last_low = swing_lows[-1][1]
        c = close[i]
        if c > last_high:
            if trend == "bear":
                last_choch = i
                trend = "bull"
            else:
                last_bos = i
                trend = "bull"
            # broken high no longer resistance; wait for a fresh swing
            swing_highs.append((i, c))
        elif c < last_low:
            if trend == "bull":
                last_choch = i
                trend = "bear"
            else:
                last_bos = i
                trend = "bear"
            swing_lows.append((i, c))

    return StructureState(
        trend=trend,
        last_bos_index=last_bos,
        last_choch_index=last_choch,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
    )
