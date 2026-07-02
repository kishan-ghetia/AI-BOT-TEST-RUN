"""Order blocks: last opposing candle before an impulsive move."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OrderBlock:
    index: int
    kind: str  # "bullish" (demand) | "bearish" (supply)
    top: float
    bottom: float


def find_order_blocks(
    df: pd.DataFrame, impulse_mult: float = 2.0, window: int = 20
) -> list[OrderBlock]:
    """A bearish (down) candle immediately followed by an up-move whose body is
    >= impulse_mult * average body => bullish order block (and vice versa)."""
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    avg_body = pd.Series(np.abs(close - open_)).rolling(window).mean().to_numpy()
    blocks: list[OrderBlock] = []
    for i in range(window, len(df) - 1):
        threshold = impulse_mult * avg_body[i]
        if not threshold > 0:  # covers nan and <= 0
            continue
        nxt_body = close[i + 1] - open_[i + 1]
        if close[i] < open_[i] and nxt_body >= threshold:
            blocks.append(OrderBlock(i, "bullish", top=high[i], bottom=low[i]))
        elif close[i] > open_[i] and -nxt_body >= threshold:
            blocks.append(OrderBlock(i, "bearish", top=high[i], bottom=low[i]))
    return blocks


def price_in_block(price: float, block: OrderBlock) -> bool:
    return block.bottom <= price <= block.top
