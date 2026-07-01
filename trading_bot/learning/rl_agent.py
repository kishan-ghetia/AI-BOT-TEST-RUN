"""Lightweight RL agent: Q-learning with linear function approximation
(one SGDRegressor per action, updated online via partial_fit).

Learns POSITION SIZING and EXIT TIMING on top of the ensemble's direction:
- entry: choose a sizing multiplier applied to the risk manager's base size
- open position: choose hold vs close each cycle

Reward: risk-adjusted realized P&L on trade close.
"""
from __future__ import annotations

import random

import numpy as np
from sklearn.linear_model import SGDRegressor

SIZE_ACTIONS = ["size_0.0x", "size_0.5x", "size_1.0x", "size_1.5x"]
EXIT_ACTIONS = ["hold", "close"]
ALL_ACTIONS = SIZE_ACTIONS + EXIT_ACTIONS

SIZE_MULTIPLIERS = {"size_0.0x": 0.0, "size_0.5x": 0.5, "size_1.0x": 1.0, "size_1.5x": 1.5}

N_FEATURES = 6  # keep in sync with state_vector


def state_vector(ensemble_value: float, ensemble_confidence: float,
                 atr_pct: float, unrealized_pnl_pct: float,
                 time_in_trade: float, trend_strength: float) -> np.ndarray:
    return np.array([
        ensemble_value,
        ensemble_confidence,
        min(atr_pct * 100.0, 5.0),          # cap extreme vol
        max(-5.0, min(5.0, unrealized_pnl_pct * 100.0)),
        min(time_in_trade / 20.0, 2.0),     # bars in trade, scaled
        max(-1.0, min(1.0, trend_strength)),
    ], dtype=float)


class RLAgent:
    def __init__(self, alpha: float = 0.01, gamma: float = 0.9,
                 epsilon: float = 0.1, seed: int | None = None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.updates = 0
        self.models: dict[str, SGDRegressor] = {}
        for action in ALL_ACTIONS:
            m = SGDRegressor(learning_rate="constant", eta0=alpha, random_state=42)
            # prime with a zero-target fit so predict() works immediately
            m.partial_fit(np.zeros((1, N_FEATURES)), np.zeros(1))
            self.models[action] = m

    def q(self, state: np.ndarray, action: str) -> float:
        return float(self.models[action].predict(state.reshape(1, -1))[0])

    def choose_action(self, state: np.ndarray, actions: list[str],
                      greedy: bool = False) -> str:
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.choice(actions)
        qs = {a: self.q(state, a) for a in actions}
        return max(qs, key=qs.get)

    def choose_size(self, state: np.ndarray, greedy: bool = False) -> tuple[str, float]:
        action = self.choose_action(state, SIZE_ACTIONS, greedy)
        return action, SIZE_MULTIPLIERS[action]

    def choose_exit(self, state: np.ndarray, greedy: bool = False) -> str:
        return self.choose_action(state, EXIT_ACTIONS, greedy)

    def update(self, state: np.ndarray, action: str, reward: float,
               next_state: np.ndarray | None, done: bool) -> None:
        """One TD(0) step: target = r + gamma * max_a' Q(s', a')."""
        target = reward
        if not done and next_state is not None:
            future = max(self.q(next_state, a) for a in ALL_ACTIONS)
            target += self.gamma * future
        self.models[action].partial_fit(state.reshape(1, -1), np.array([target]))
        self.updates += 1

    # --- persistence ---
    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "updates": self.updates,
            "coefs": {a: m.coef_.tolist() for a, m in self.models.items()},
            "intercepts": {a: float(m.intercept_[0]) for a, m in self.models.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RLAgent":
        agent = cls(alpha=data.get("alpha", 0.01), gamma=data.get("gamma", 0.9),
                    epsilon=data.get("epsilon", 0.1))
        agent.updates = data.get("updates", 0)
        for action, coefs in data.get("coefs", {}).items():
            if action in agent.models:
                agent.models[action].coef_ = np.asarray(coefs, dtype=float)
                agent.models[action].intercept_ = np.array(
                    [data["intercepts"][action]], dtype=float)
        return agent


def trade_reward(realized_pnl_pct: float, atr_pct_at_entry: float) -> float:
    """Risk-adjusted reward: P&L normalized by volatility at entry."""
    return realized_pnl_pct / (atr_pct_at_entry + 1e-6)
