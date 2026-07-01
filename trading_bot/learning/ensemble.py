"""Ensemble: combine per-strategy signals into one, weighted by the adaptive
weights; report per-strategy rewards back to the weight learner."""
from __future__ import annotations

import pandas as pd

from trading_bot.learning.adaptive_weights import AdaptiveWeights
from trading_bot.strategies.base import Signal, Strategy


class Ensemble:
    def __init__(self, strategies: list[Strategy], weights: AdaptiveWeights):
        self.strategies = strategies
        self.weights = weights
        self.weights.sync_strategies([s.name for s in strategies])
        self._last_components: list[Signal] = []

    def combine(self, df: pd.DataFrame, context: dict) -> Signal:
        components: list[Signal] = []
        for strat in self.strategies:
            try:
                components.append(strat.generate_signal(df, context))
            except Exception as exc:  # one broken trader must not sink the desk
                import logging

                logging.getLogger(__name__).warning(
                    "Strategy %s failed: %s", strat.name, exc)
        self._last_components = components

        num = 0.0
        den = 0.0
        for sig in components:
            w = self.weights.get(sig.strategy_name) * sig.confidence
            num += w * sig.value
            den += w
        value = num / den if den > 0 else 0.0
        confidence = min(1.0, den)
        return Signal(
            symbol=context.get("symbol", "?"),
            timeframe=context.get("timeframe", "?"),
            timestamp=df.index[-1] if len(df) else None,
            strategy_name="ensemble",
            value=value,
            confidence=confidence,
            metadata={
                "components": {
                    s.strategy_name: {"value": s.value, "confidence": s.confidence}
                    for s in components
                },
                "weights": dict(self.weights.weights),
            },
        ).clamped()

    def update(self, realized_return: float) -> dict[str, float]:
        """Feed realized return back: reward_i = signal_i * realized_return
        (scaled so a 1% move with full conviction ~= reward 1)."""
        rewards = {
            sig.strategy_name: sig.value * realized_return * 100.0
            for sig in self._last_components
        }
        self.weights.update(rewards)
        return rewards
