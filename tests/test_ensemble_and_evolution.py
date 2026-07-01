from pathlib import Path

from tests.conftest import make_ohlcv
from trading_bot.learning.adaptive_weights import AdaptiveWeights
from trading_bot.learning.ensemble import Ensemble
from trading_bot.learning.evolution import StrategyPool, mutate_params, GENE_SPACE
from trading_bot.strategies.base import Signal, Strategy


class Stub(Strategy):
    def __init__(self, name, value, confidence=1.0):
        self.name = name
        self._v = value
        self._c = confidence

    def generate_signal(self, df, context):
        return Signal("T", "daily", df.index[-1], self.name, self._v, self._c)


def test_ensemble_weighted_combination(choppy):
    bull = Stub("bull", 1.0)
    bear = Stub("bear", -1.0)
    w = AdaptiveWeights(["bull", "bear"])
    w.weights = {"bull": 0.75, "bear": 0.25}
    ens = Ensemble([bull, bear], w)
    sig = ens.combine(choppy, {"symbol": "T", "timeframe": "daily"})
    # 0.75*1 + 0.25*(-1) = 0.5
    assert abs(sig.value - 0.5) < 1e-9


def test_ensemble_confidence_weighting(choppy):
    confident = Stub("confident", 1.0, confidence=1.0)
    unsure = Stub("unsure", -1.0, confidence=0.1)
    ens = Ensemble([confident, unsure],
                   AdaptiveWeights(["confident", "unsure"]))
    sig = ens.combine(choppy, {"symbol": "T", "timeframe": "daily"})
    assert sig.value > 0.5  # confident long dominates


def test_ensemble_update_rewards_correct_side(choppy):
    bull = Stub("bull", 1.0)
    bear = Stub("bear", -1.0)
    ens = Ensemble([bull, bear], AdaptiveWeights(["bull", "bear"]))
    ens.combine(choppy, {"symbol": "T", "timeframe": "daily"})
    ens.update(realized_return=0.02)  # market went up
    assert ens.weights.get("bull") > ens.weights.get("bear")


def test_ensemble_survives_broken_strategy(choppy):
    class Broken(Strategy):
        name = "broken"

        def generate_signal(self, df, context):
            raise RuntimeError("boom")

    ens = Ensemble([Stub("ok", 0.5), Broken()],
                   AdaptiveWeights(["ok", "broken"]))
    sig = ens.combine(choppy, {"symbol": "T", "timeframe": "daily"})
    assert sig.value > 0  # still produced a signal from the healthy trader


# --- evolution ---

def test_pool_seeds_and_persists(tmp_path):
    pool = StrategyPool("daily", tmp_path, pool_size=6, seed=1)
    assert len(pool.traders) == 6
    reloaded = StrategyPool("daily", tmp_path, pool_size=6, seed=2)
    assert [t.trader_id for t in reloaded.traders] == \
           [t.trader_id for t in pool.traders]


def test_evolution_fires_worst_and_hires(tmp_path):
    pool = StrategyPool("daily", tmp_path, pool_size=6, cull_fraction=0.34, seed=1)
    fitness = {t.trader_id: float(i) for i, t in enumerate(pool.traders)}
    worst = pool.traders[0].trader_id  # fitness 0
    pool.record_fitness(fitness)
    report = pool.evolve()
    ids = [t.trader_id for t in pool.traders]
    assert worst in report["culled"]
    assert worst not in ids
    assert len(ids) == 6  # pool size maintained
    assert report["generation"] == 1
    assert len(report["spawned"]) == len(report["culled"])


def test_mutation_stays_in_bounds():
    import random

    rng = random.Random(7)
    params = {"fast": 10, "slow": 50, "rsi_window": 14}
    for _ in range(50):
        params = mutate_params("technical", params, rng)
        for name, (lo, hi, _) in GENE_SPACE["technical"].items():
            assert lo <= params[name] <= hi
        assert params["fast"] < params["slow"]


def test_traders_build_runnable_strategies(tmp_path, choppy):
    pool = StrategyPool("daily", tmp_path, pool_size=6, seed=3)
    for trader in pool.traders:
        strat = trader.build()
        sig = strat.generate_signal(choppy, {"symbol": "T", "timeframe": "daily"})
        assert -1.0 <= sig.value <= 1.0
