"""Synthetic OHLCV generation: regime-switching geometric Brownian motion.

Used when real market data is unreachable (offline environments, CI) and as
a training gym for the evolution loop. Deterministic per (symbol, timeframe).
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

BARS = {"daily": 730, "hourly": 2000, "m5": 3000}
FREQ = {"daily": "D", "hourly": "h", "m5": "5min"}


def generate_ohlcv(symbol: str, timeframe: str = "daily",
                   n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Regime-switching random walk: alternating bull/bear/chop segments so
    trend-following and mean-reversion strategies both get exercised."""
    n = n or BARS[timeframe]
    if seed is None:
        seed = zlib.crc32(f"{symbol}:{timeframe}".encode()) & 0xFFFF
    rng = np.random.default_rng(seed)

    rets = np.empty(n)
    i = 0
    while i < n:
        seg = int(rng.integers(30, 120))
        regime = rng.choice(["bull", "bear", "chop"], p=[0.4, 0.25, 0.35])
        drift = {"bull": 0.0015, "bear": -0.0015, "chop": 0.0}[regime]
        vol = {"bull": 0.012, "bear": 0.02, "chop": 0.008}[regime]
        take = min(seg, n - i)
        rets[i : i + take] = drift + vol * rng.standard_normal(take)
        i += take

    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.standard_normal(n)) * 0.004 * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(1e5, 1e6, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=FREQ[timeframe])
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)
