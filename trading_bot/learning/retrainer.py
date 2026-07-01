"""Periodic ML model retraining on a rolling window."""
from __future__ import annotations

import logging

import pandas as pd

from trading_bot.archive.store import Archive
from trading_bot.strategies.ml_strategy import train_model

logger = logging.getLogger(__name__)

# retrain every N cycles per timeframe
RETRAIN_EVERY = {"daily": 5, "hourly": 24, "m5": 96}
ROLLING_WINDOW_BARS = {"daily": 500, "hourly": 1000, "m5": 2000}


def maybe_retrain(archive: Archive, df: pd.DataFrame, cycle_count: int) -> bool:
    """Retrain if the cadence says so. Returns True when a new model was saved."""
    every = RETRAIN_EVERY.get(archive.timeframe, 10)
    if cycle_count % every != 0:
        return False
    window = ROLLING_WINDOW_BARS.get(archive.timeframe, 500)
    model = train_model(df.tail(window))
    if model is None:
        logger.info("Retrain skipped for %s/%s: insufficient data",
                    archive.timeframe, archive.symbol)
        return False
    archive.save_model(model)
    logger.info("Retrained ML model for %s/%s", archive.timeframe, archive.symbol)
    return True
