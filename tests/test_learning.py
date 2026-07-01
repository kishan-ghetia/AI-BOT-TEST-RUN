import numpy as np

from trading_bot.learning.adaptive_weights import AdaptiveWeights
from trading_bot.learning.rl_agent import (
    RLAgent, SIZE_ACTIONS, state_vector, trade_reward,
)


# --- adaptive weights ---

def test_weights_start_uniform_and_normalized():
    w = AdaptiveWeights(["a", "b", "c", "d"])
    assert abs(sum(w.weights.values()) - 1.0) < 1e-9
    assert abs(w.get("a") - 0.25) < 1e-9


def test_positive_reward_gains_weight():
    w = AdaptiveWeights(["good", "bad"])
    for _ in range(10):
        w.update({"good": 0.5, "bad": -0.5})
    assert w.get("good") > w.get("bad")
    assert abs(sum(w.weights.values()) - 1.0) < 1e-9


def test_floor_prevents_silencing():
    w = AdaptiveWeights(["good", "bad"], floor=0.02)
    for _ in range(200):
        w.update({"good": 1.0, "bad": -1.0})
    assert w.get("bad") >= 0.015  # floor holds (post-normalization)


def test_round_trip_persistence():
    w = AdaptiveWeights(["a", "b"])
    w.update({"a": 0.7, "b": -0.3})
    restored = AdaptiveWeights.from_dict(w.to_dict())
    assert restored.weights == w.weights


def test_sync_adds_and_removes():
    w = AdaptiveWeights(["a", "b"])
    w.sync_strategies(["a", "c"])
    assert "c" in w.weights and "b" not in w.weights


# --- RL agent ---

def test_rl_converges_to_rewarded_action():
    agent = RLAgent(alpha=0.05, epsilon=0.0, seed=1)
    state = state_vector(0.8, 0.9, 0.02, 0.0, 0.0, 0.8)
    for _ in range(60):
        agent.update(state, "size_1.5x", reward=1.0, next_state=None, done=True)
        agent.update(state, "size_0.0x", reward=-1.0, next_state=None, done=True)
    assert agent.choose_size(state, greedy=True)[0] == "size_1.5x"


def test_rl_round_trip():
    agent = RLAgent(seed=2)
    state = state_vector(0.5, 0.5, 0.01, 0.01, 3, 0.5)
    for _ in range(20):
        agent.update(state, "close", 0.7, None, True)
    restored = RLAgent.from_dict(agent.to_dict())
    for action in SIZE_ACTIONS + ["close", "hold"]:
        assert abs(agent.q(state, action) - restored.q(state, action)) < 1e-9


def test_trade_reward_risk_adjusts():
    # same pnl, higher vol at entry => lower reward
    assert trade_reward(0.02, 0.01) > trade_reward(0.02, 0.05)


def test_state_vector_caps():
    st = state_vector(2.0, 1.0, 10.0, 99.0, 1000, 5.0)
    assert st[2] <= 5.0 and st[3] <= 5.0 and st[5] <= 1.0
    assert np.isfinite(st).all()
