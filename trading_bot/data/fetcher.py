"""Market data fetching via yfinance, normalized to a standard OHLCV frame."""
from __future__ import annotations

import pandas as pd

# yfinance interval / default lookback per bot timeframe
TIMEFRAME_MAP = {
    "daily": {"interval": "1d", "period": "2y"},
    "hourly": {"interval": "1h", "period": "180d"},
    "m5": {"interval": "5m", "period": "30d"},
}

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, flatten MultiIndex, keep OHLCV, drop bad rows."""
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    df = df[[c for c in REQUIRED_COLS if c in df.columns]].copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[REQUIRED_COLS].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index)
    return df


def get_ohlcv(
    symbol: str,
    timeframe: str = "daily",
    start: str | None = None,
    end: str | None = None,
    synthetic: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV bars for a symbol. Network call - not used in tests.

    With synthetic=True (or SYNTHETIC_FALLBACK=true in .env when the real
    fetch fails) returns deterministic simulated data instead."""
    import logging

    import config
    from trading_bot.data.synthetic import generate_ohlcv

    if synthetic:
        return generate_ohlcv(symbol, timeframe)

    import yfinance as yf

    spec = TIMEFRAME_MAP[timeframe]
    kwargs = dict(interval=spec["interval"], auto_adjust=True, progress=False)
    if start:
        kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = spec["period"]
    try:
        raw = yf.download(symbol, **kwargs)
    except Exception:
        raw = None
    df = normalize_ohlcv(raw)
    if df.empty and config.SYNTHETIC_FALLBACK:
        logging.getLogger(__name__).warning(
            "REAL DATA UNAVAILABLE for %s - using SYNTHETIC data. "
            "Results are for pipeline testing only.", symbol)
        return generate_ohlcv(symbol, timeframe)
    return df
