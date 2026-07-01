from datetime import datetime, timedelta, timezone

import config
from tests.conftest import make_ohlcv
from trading_bot.archive.store import Archive
from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.learning.adaptive_weights import AdaptiveWeights
from trading_bot.learning.ensemble import Ensemble
from trading_bot.learning.rl_agent import RLAgent
from trading_bot.strategies.base import Signal, Strategy
from trading_bot.strategies.technical_strategy import TechnicalStrategy


class AlwaysLong(Strategy):
    name = "always_long"

    def generate_signal(self, df, context):
        return Signal(context.get("symbol", "?"), context.get("timeframe", "?"),
                      df.index[-1], self.name, 1.0, 1.0)


def test_backtest_runs_and_produces_metrics(trending_up):
    ens = Ensemble([TechnicalStrategy()], AdaptiveWeights(["technical"]))
    res = run_backtest(trending_up, ens, "TEST", "daily", rl_agent=RLAgent(seed=1))
    assert len(res.equity_curve) > 50
    assert "sharpe" in res.metrics and "max_drawdown" in res.metrics
    assert res.metrics["trade_count"] == len(res.trades)


def test_always_long_profits_in_uptrend(trending_up):
    ens = Ensemble([AlwaysLong()], AdaptiveWeights(["always_long"]))
    res = run_backtest(trending_up, ens, "TEST", "daily")
    assert res.metrics["total_return"] > 0


def test_drawdown_breaker_halts(trending_down):
    ens = Ensemble([AlwaysLong()], AdaptiveWeights(["always_long"]))
    res = run_backtest(trending_down, ens, "TEST", "daily")
    # equity must never fall much past the 20% breaker level
    assert min(res.equity_curve) > config.INITIAL_CAPITAL * 0.7


def test_metrics_hand_checkable():
    m = compute_metrics([100, 110, 105, 120], "daily",
                        trades=[{"pnl": 5}, {"pnl": -2}, {"pnl": 3}])
    assert abs(m["total_return"] - 0.20) < 1e-9
    assert m["win_rate"] == 2 / 3
    assert m["max_drawdown"] > 0


def test_archive_round_trip(temp_store):
    a = Archive(temp_store, "daily", "TEST")
    w = AdaptiveWeights(["x", "y"])
    w.update({"x": 0.5, "y": -0.5})
    a.save_weights_dict(w.to_dict())
    assert AdaptiveWeights.from_dict(a.load_weights_dict()).weights == w.weights

    agent = RLAgent(seed=3)
    a.save_rl_dict(agent.to_dict())
    assert a.load_rl_dict()["updates"] == 0

    a.append_trade({"timestamp": "2024-01-01", "symbol": "TEST", "side": "long",
                    "size": 1, "entry_price": 100, "exit_price": 105,
                    "pnl": 5, "pnl_pct": 0.05, "rl_action": "size_1.0x",
                    "ensemble_signal": 0.5})
    assert a.load_performance()["trade_count"] == 1


def test_archive_continuity_reset(temp_store):
    a = Archive(temp_store, "hourly", "TEST")
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    a.update_performance(10_000, t0, cadence_seconds=3600)
    a.update_performance(10_100, t0 + timedelta(hours=1), cadence_seconds=3600)
    perf = a.load_performance()
    assert perf["started_at"] == t0.isoformat()
    # a 3-day gap on an hourly cadence resets continuity
    perf = a.update_performance(10_200, t0 + timedelta(days=3), cadence_seconds=3600)
    assert perf["started_at"] == (t0 + timedelta(days=3)).isoformat()
    assert len(perf["equity_curve"]) == 1
