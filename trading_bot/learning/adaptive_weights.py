"""Multiplicative-weights (Hedge) online learner for ensemble strategy weights.

Reward for a strategy = its signal value * realized return: strategies that
called the direction right (with conviction) gain weight; wrong ones lose it.
"""
from __future__ import annotations

import math


class AdaptiveWeights:
    def __init__(self, strategy_names: list[str], eta: float = 0.1,
                 floor: float = 0.02):
        self.eta = eta
        self.floor = floor  # no strategy is ever fully silenced
        self.weights = {name: 1.0 / len(strategy_names) for name in strategy_names}

    def get(self, name: str) -> float:
        return self.weights.get(name, 0.0)

    def update(self, rewards: dict[str, float]) -> None:
        """rewards: strategy_name -> signed reward (clipped to [-1, 1])."""
        for name, reward in rewards.items():
            if name in self.weights:
                r = max(-1.0, min(1.0, reward))
                self.weights[name] *= math.exp(self.eta * r)
        self._normalize()

    def _normalize(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            n = len(self.weights)
            self.weights = {k: 1.0 / n for k in self.weights}
            return
        self.weights = {k: v / total for k, v in self.weights.items()}
        # apply floor then renormalize once more
        floored = {k: max(v, self.floor) for k, v in self.weights.items()}
        total = sum(floored.values())
        self.weights = {k: v / total for k, v in floored.items()}

    def to_dict(self) -> dict:
        return {"eta": self.eta, "floor": self.floor, "weights": dict(self.weights)}

    @classmethod
    def from_dict(cls, data: dict) -> "AdaptiveWeights":
        names = list(data.get("weights", {}).keys()) or ["technical"]
        obj = cls(names, eta=data.get("eta", 0.1), floor=data.get("floor", 0.02))
        obj.weights = dict(data["weights"])
        return obj

    def sync_strategies(self, strategy_names: list[str]) -> None:
        """Add newly-registered strategies at average weight; drop removed ones."""
        avg = sum(self.weights.values()) / max(len(self.weights), 1)
        for name in strategy_names:
            self.weights.setdefault(name, avg)
        for name in list(self.weights):
            if name not in strategy_names:
                del self.weights[name]
        self._normalize()
