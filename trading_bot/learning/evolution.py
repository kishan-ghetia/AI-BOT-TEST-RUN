"""Evolutionary strategy pool: the "trading floor" model.

N parameterized strategy variants ("traders") compete. Each generation:
fitness = backtest performance; the bottom performers are fired, the top
performers stay and spawn mutated offspring. Survivors join the live
ensemble, where adaptive weights keep judging them online.

Population state persists to data_store/evolution/<timeframe>/population.json.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from trading_bot.strategies.base import Strategy
from trading_bot.strategies.fibonacci import FibonacciStrategy
from trading_bot.strategies.smc_ict.strategy import SMCICTStrategy
from trading_bot.strategies.technical_strategy import TechnicalStrategy

# gene space: strategy kind -> {param: (low, high, is_int)}
GENE_SPACE = {
    "technical": {"fast": (5, 20, True), "slow": (30, 100, True),
                  "rsi_window": (7, 21, True)},
    "fibonacci": {"lookback": (2, 6, True)},
    "smc_ict": {"lookback": (2, 6, True), "recent_bars": (5, 20, True)},
}

FACTORIES = {
    "technical": TechnicalStrategy,
    "fibonacci": FibonacciStrategy,
    "smc_ict": SMCICTStrategy,
}


def validation_fitness(equity_curve: list[float], timeframe: str = "daily",
                       validation_frac: float = 0.3,
                       dd_penalty: float = 2.0) -> float:
    """Walk-forward fitness: judge a trader only on the OUT-OF-SAMPLE tail of
    its backtest (the learners adapted on the earlier bars). Sharpe on the
    validation segment minus a drawdown penalty."""
    from trading_bot.backtest.metrics import compute_metrics

    n = len(equity_curve)
    if n < 10:
        return -10.0  # not enough evidence: worst possible desk
    tail = equity_curve[int(n * (1 - validation_frac)):]
    m = compute_metrics(tail, timeframe)
    return m["sharpe"] - dd_penalty * m["max_drawdown"]


@dataclass
class Trader:
    trader_id: str
    kind: str
    params: dict
    fitness: float | None = None
    generations_survived: int = 0
    history: list = field(default_factory=list)

    def build(self) -> Strategy:
        strat = FACTORIES[self.kind](**self.params)
        strat.name = self.trader_id  # unique name so weights track each trader
        return strat


def random_params(kind: str, rng: random.Random) -> dict:
    params = {}
    for name, (lo, hi, is_int) in GENE_SPACE[kind].items():
        params[name] = rng.randint(lo, hi) if is_int else rng.uniform(lo, hi)
    if kind == "technical" and params["fast"] >= params["slow"]:
        params["fast"] = max(2, params["slow"] // 3)
    return params


def mutate_params(kind: str, params: dict, rng: random.Random,
                  scale: float = 0.25) -> dict:
    out = {}
    for name, (lo, hi, is_int) in GENE_SPACE[kind].items():
        val = params[name]
        span = (hi - lo) * scale
        newval = val + rng.uniform(-span, span)
        newval = max(lo, min(hi, newval))
        out[name] = int(round(newval)) if is_int else newval
    if kind == "technical" and out["fast"] >= out["slow"]:
        out["fast"] = max(2, out["slow"] // 3)
    return out


class StrategyPool:
    def __init__(self, timeframe: str, base_dir: Path, pool_size: int = 12,
                 cull_fraction: float = 0.33, seed: int | None = None):
        self.timeframe = timeframe
        self.path = Path(base_dir) / "evolution" / timeframe / "population.json"
        self.pool_size = pool_size
        self.cull_fraction = cull_fraction
        self.rng = random.Random(seed)
        self.generation = 0
        self.traders: list[Trader] = []
        self._load()
        if not self.traders:
            self._seed_population()

    def _seed_population(self) -> None:
        kinds = list(GENE_SPACE.keys())
        for i in range(self.pool_size):
            kind = kinds[i % len(kinds)]
            self.traders.append(Trader(
                trader_id=f"{kind}_g0_{i}",
                kind=kind,
                params=random_params(kind, self.rng),
            ))
        self._save()

    def build_strategies(self) -> list[Strategy]:
        return [t.build() for t in self.traders]

    def record_fitness(self, fitness_by_id: dict[str, float]) -> None:
        for t in self.traders:
            if t.trader_id in fitness_by_id:
                t.fitness = fitness_by_id[t.trader_id]
                t.history.append(t.fitness)

    def evolve(self) -> dict:
        """Fire the bottom, keep the top, spawn mutated offspring of winners.

        Diversity protection: the best trader of each kind is immune from
        culling (styles go cold, not extinct - regimes rotate), and any kind
        missing from the pool gets a fresh random hire."""
        scored = [t for t in self.traders if t.fitness is not None]
        if len(scored) < 2:
            return {"culled": [], "spawned": [], "generation": self.generation}
        scored.sort(key=lambda t: t.fitness, reverse=True)

        protected: set[str] = set()
        seen_kinds: set[str] = set()
        for t in scored:  # best of each kind, in fitness order
            if t.kind not in seen_kinds:
                protected.add(t.trader_id)
                seen_kinds.add(t.kind)

        n_cull = max(1, int(len(scored) * self.cull_fraction))
        cullable = [t for t in reversed(scored) if t.trader_id not in protected]
        culled = cullable[:n_cull]
        culled_ids = {t.trader_id for t in culled}
        survivors = [t for t in scored if t.trader_id not in culled_ids]
        for t in survivors:
            t.generations_survived += 1

        self.generation += 1
        spawned: list[Trader] = []
        parents = survivors[: max(1, len(survivors) // 2)]  # elite breed
        surviving_kinds = {t.kind for t in survivors}
        missing_kinds = [k for k in GENE_SPACE if k not in surviving_kinds]
        for i in range(len(culled)):
            if missing_kinds:  # re-open the extinct desk with a fresh hire
                kind = missing_kinds.pop(0)
                child = Trader(
                    trader_id=f"{kind}_g{self.generation}_{i}",
                    kind=kind,
                    params=random_params(kind, self.rng),
                )
            else:
                parent = parents[i % len(parents)]
                child = Trader(
                    trader_id=f"{parent.kind}_g{self.generation}_{i}",
                    kind=parent.kind,
                    params=mutate_params(parent.kind, parent.params, self.rng),
                )
            spawned.append(child)

        self.traders = survivors + spawned
        self._save()
        return {
            "culled": [t.trader_id for t in culled],
            "spawned": [t.trader_id for t in spawned],
            "generation": self.generation,
            "best": survivors[0].trader_id if survivors else None,
            "best_fitness": survivors[0].fitness if survivors else None,
        }

    # --- persistence ---
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generation": self.generation,
            "pool_size": self.pool_size,
            "traders": [
                {"trader_id": t.trader_id, "kind": t.kind, "params": t.params,
                 "fitness": t.fitness,
                 "generations_survived": t.generations_survived,
                 "history": t.history[-20:]}
                for t in self.traders
            ],
        }
        self.path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        self.generation = data.get("generation", 0)
        self.pool_size = data.get("pool_size", self.pool_size)
        self.traders = [Trader(**t) for t in data.get("traders", [])]
