"""ML strategy: gradient-boosting classifier predicting next-bar direction.

Model lives in the per-timeframe archive; if none exists yet the strategy
fits an initial model on the available history (so it can produce signals
before the retrainer has ever run).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from trading_bot.data.features import FEATURE_COLUMNS, build_features, build_labels
from trading_bot.strategies.base import Signal, Strategy

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 100


def train_model(df: pd.DataFrame) -> GradientBoostingClassifier | None:
    feats = build_features(df)
    labels = build_labels(df).reindex(feats.index)
    mask = labels.notna()
    X, y = feats[mask], labels[mask]
    # drop the final row(s) whose label peeks past available data
    X, y = X.iloc[:-1], y.iloc[:-1]
    if len(X) < MIN_TRAIN_ROWS or y.nunique() < 2:
        return None
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X[FEATURE_COLUMNS], y)
    return model


class MLStrategy(Strategy):
    name = "ml"

    def __init__(self, model=None):
        self.model = model

    def warmup_bars(self) -> int:
        return MIN_TRAIN_ROWS + 30

    def generate_signal(self, df: pd.DataFrame, context: dict) -> Signal:
        archive = context.get("archive")
        if self.model is None and archive is not None:
            self.model = archive.load_model()
        if self.model is None:
            self.model = train_model(df)
            if self.model is not None and archive is not None:
                archive.save_model(self.model)
        if self.model is None:
            return self.neutral(df, context, reason="insufficient_history")

        feats = build_features(df)
        if feats.empty:
            return self.neutral(df, context, reason="no_features")
        x = feats[FEATURE_COLUMNS].iloc[[-1]]
        try:
            proba_up = float(self.model.predict_proba(x)[0][1])
        except Exception as exc:  # stale model vs feature drift
            logger.warning("ML predict failed, retraining: %s", exc)
            self.model = train_model(df)
            if self.model is None:
                return self.neutral(df, context, reason="retrain_failed")
            proba_up = float(self.model.predict_proba(x)[0][1])

        value = 2.0 * proba_up - 1.0  # map [0,1] -> [-1,1]
        confidence = abs(value)  # distance from coin-flip
        return Signal(
            symbol=context.get("symbol", "?"),
            timeframe=context.get("timeframe", "?"),
            timestamp=df.index[-1],
            strategy_name=self.name,
            value=value,
            confidence=confidence,
            metadata={"proba_up": proba_up},
        ).clamped()
