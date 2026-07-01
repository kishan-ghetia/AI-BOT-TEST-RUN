"""Feature engineering for the ML strategy."""
from __future__ import annotations

import pandas as pd

from trading_bot.strategies import indicators as ta

FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "rsi",
    "macd_hist",
    "bb_pos",
    "atr_pct",
    "mom_20",
    "vol_20",
    "ema_ratio",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model features from OHLCV. Rows with NaNs are dropped."""
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["rsi"] = ta.rsi(close) / 100.0
    _, _, hist = ta.macd(close)
    out["macd_hist"] = hist / close
    upper, mid, lower = ta.bollinger(close)
    width = (upper - lower).replace(0, pd.NA)
    out["bb_pos"] = ((close - lower) / width).astype(float)
    out["atr_pct"] = ta.atr(df) / close
    out["mom_20"] = close.pct_change(20)
    out["vol_20"] = close.pct_change().rolling(20).std()
    out["ema_ratio"] = ta.ema(close, 10) / ta.ema(close, 50) - 1.0
    return out.dropna()


def build_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Label = 1 if next-`horizon` return positive, else 0."""
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int)
