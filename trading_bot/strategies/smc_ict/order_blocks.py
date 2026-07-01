"""Order blocks: last opposing candle before an impulsive move."""
from __future__ import annotations

from dataclasses import dataclass

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
    body = (df["close"] - df["open"]).abs()
    avg_body = body.rolling(window).mean()
    blocks: list[OrderBlock] = []
    for i in range(window, len(df) - 1):
        nxt_body = df["close"].iloc[i + 1] - df["open"].iloc[i + 1]
        threshold = impulse_mult * avg_body.iloc[i]
        if threshold <= 0 or pd.isna(threshold):
            continue
        is_down = df["close"].iloc[i] < df["open"].iloc[i]
        is_up = df["close"].iloc[i] > df["open"].iloc[i]
        if is_down and nxt_body >= threshold:
            blocks.append(OrderBlock(i, "bullish",
                                     top=df["high"].iloc[i],
                                     bottom=df["low"].iloc[i]))
        elif is_up and -nxt_body >= threshold:
            blocks.append(OrderBlock(i, "bearish",
                                     top=df["high"].iloc[i],
                                     bottom=df["low"].iloc[i]))
    return blocks


def price_in_block(price: float, block: OrderBlock) -> bool:
    return block.bottom <= price <= block.top
