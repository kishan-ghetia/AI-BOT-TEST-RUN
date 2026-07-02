"""Shared technical-indicator math (pure pandas/numpy, no TA-lib dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss  # loss==0 with gains -> inf -> RSI 100
        out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def swing_points(df: pd.DataFrame, lookback: int = 3):
    """Return (swing_high_mask, swing_low_mask): bar is a local extreme vs
    `lookback` bars on each side. Vectorized (centered rolling extremes)."""
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    win = 2 * lookback + 1
    if n >= win:
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        hmax = pd.Series(high).rolling(win, center=True).max().to_numpy()
        lmin = pd.Series(low).rolling(win, center=True).min().to_numpy()
        core = slice(lookback, n - lookback)
        sh[core] = high[core] == hmax[core]
        sl[core] = low[core] == lmin[core]
    return pd.Series(sh, index=df.index), pd.Series(sl, index=df.index)
