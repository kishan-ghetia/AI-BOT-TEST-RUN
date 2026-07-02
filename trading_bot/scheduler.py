"""Scheduler: each timeframe runs on its own cadence; a cycle fetches data,
asks the ensemble, sizes with risk manager + RL, executes via the broker,
then persists everything back to the per-timeframe archive."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
from trading_bot.archive.store import Archive
from trading_bot.data.fetcher import get_ohlcv
from trading_bot.data.finnhub_client import FinnhubClient
from trading_bot.execution.broker_base import Broker, Order
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.learning.adaptive_weights import AdaptiveWeights
from trading_bot.learning.ensemble import Ensemble
from trading_bot.learning.retrainer import maybe_retrain
from trading_bot.learning.rl_agent import RLAgent, state_vector, trade_reward
from trading_bot.risk.manager import RiskManager
from trading_bot.risk.news_filter import risk_multiplier
from trading_bot.strategies import indicators as ta
from trading_bot.strategies.fibonacci import FibonacciStrategy
from trading_bot.strategies.ml_strategy import MLStrategy
from trading_bot.strategies.news_strategy import NewsStrategy
from trading_bot.strategies.smc_ict.strategy import SMCICTStrategy
from trading_bot.strategies.technical_strategy import TechnicalStrategy

logger = logging.getLogger(__name__)

CADENCE_SECONDS = {"daily": 86_400, "hourly": 3_600, "m5": 300}
ENTRY_THRESHOLD = 0.15


def default_strategies(finnhub: FinnhubClient) -> list:
    return [
        TechnicalStrategy(),
        MLStrategy(),
        SMCICTStrategy(),
        FibonacciStrategy(),
        NewsStrategy(finnhub),
    ]


class Scheduler:
    def __init__(self, base_dir: Path = config.DATA_STORE_DIR,
                 broker: Broker | None = None):
        self.base_dir = Path(base_dir)
        self.broker = broker or PaperBroker()
        self.finnhub = FinnhubClient(config.FINNHUB_API_KEY)
        self.last_run: dict[str, datetime] = {}
        self.cycle_counts: dict[str, int] = {}

    def due_timeframes(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now(timezone.utc)
        due = []
        for tf, seconds in CADENCE_SECONDS.items():
            last = self.last_run.get(tf)
            if last is None or (now - last).total_seconds() >= seconds:
                due.append(tf)
        return due

    def run_cycle(self, timeframe: str, symbols: list[str],
                  now: datetime | None = None) -> list[dict]:
        """One full pass over `symbols` for one timeframe. Returns actions."""
        now = now or datetime.now(timezone.utc)
        self.last_run[timeframe] = now
        actions: list[dict] = []

        # one calendar fetch per cycle, shared across symbols
        frm = (now - timedelta(hours=12)).strftime("%Y-%m-%d")
        to = (now + timedelta(hours=12)).strftime("%Y-%m-%d")
        calendar = self.finnhub.get_economic_calendar(frm, to)
        news_mult = risk_multiplier(calendar, now)

        for symbol in symbols:
            try:
                action = self._run_symbol(timeframe, symbol, now, calendar, news_mult)
                actions.append(action)
            except Exception as exc:
                logger.exception("Cycle failed for %s/%s: %s", timeframe, symbol, exc)
                actions.append({"symbol": symbol, "action": "error", "error": str(exc)})
        return actions

    def _run_symbol(self, timeframe: str, symbol: str, now: datetime,
                    calendar: list[dict], news_mult: float) -> dict:
        df = get_ohlcv(symbol, timeframe)
        if df.empty or len(df) < 60:
            return {"symbol": symbol, "action": "skip", "reason": "no_data"}

        archive = Archive(self.base_dir, timeframe, symbol)
        key = f"{timeframe}/{symbol}"
        self.cycle_counts[key] = self.cycle_counts.get(key, 0) + 1

        # --- load learning state ---
        strategies = default_strategies(self.finnhub)
        wd = archive.load_weights_dict()
        weights = AdaptiveWeights.from_dict(wd) if wd else AdaptiveWeights(
            [s.name for s in strategies])
        ensemble = Ensemble(strategies, weights)
        rl_dict = archive.load_rl_dict()
        rl_agent = RLAgent.from_dict(rl_dict) if rl_dict else RLAgent()
        risk = RiskManager()

        price = float(df["close"].iloc[-1])
        atr_val = ta.atr(df).iloc[-1]
        atr_val = float(atr_val) if not pd.isna(atr_val) else price * 0.01
        atr_pct = atr_val / price if price else 0.01

        context = {"symbol": symbol, "timeframe": timeframe,
                   "archive": archive, "calendar_events": calendar}
        signal = ensemble.combine(df, context)

        equity = self.broker.get_equity({symbol: price})

        # per-symbol equity for THIS archive: this symbol's own realized +
        # unrealized P&L only, so one market's losses can't contaminate
        # another's gate statistics (global equity still drives sizing)
        realized_map = getattr(self.broker, "realized_by_symbol", None)
        if realized_map is not None:
            open_pos = next((p for p in self.broker.get_positions()
                             if p.symbol == symbol), None)
            unrealized = open_pos.unrealized_pnl(price) if open_pos else 0.0
            symbol_equity = (config.INITIAL_CAPITAL
                             + realized_map.get(symbol, 0.0) + unrealized)
        else:  # real brokers don't attribute per symbol; use account equity
            symbol_equity = equity
        perf = archive.update_performance(symbol_equity, now,
                                          CADENCE_SECONDS[timeframe])

        result = {"symbol": symbol, "timeframe": timeframe, "action": "hold",
                  "signal": round(signal.value, 4),
                  "confidence": round(signal.confidence, 4),
                  "news_multiplier": news_mult}

        # circuit breaker on the archive's equity history
        curve = [pt[1] for pt in perf.get("equity_curve", [])]
        if risk.check_drawdown_breaker(curve):
            result["action"] = "halted_drawdown"
            self._flatten(symbol, price, now, archive, ensemble, rl_agent, atr_pct)
            self._persist(archive, ensemble, rl_agent)
            return result

        # manage open position
        pos = next((p for p in self.broker.get_positions() if p.symbol == symbol), None)
        if pos is not None:
            pos.bars_held += 1
            st = state_vector(signal.value, signal.confidence, atr_pct,
                              pos.unrealized_pnl_pct(price), pos.bars_held,
                              signal.value)
            flip = (pos.side == "long" and signal.value < -ENTRY_THRESHOLD) or (
                pos.side == "short" and signal.value > ENTRY_THRESHOLD)
            if rl_agent.choose_exit(st) == "close" or flip:
                fill = self.broker.close_position(symbol, price, now)
                self._learn_from_close(fill, archive, ensemble, rl_agent, atr_pct, now)
                result["action"] = "closed"
        elif abs(signal.value) >= ENTRY_THRESHOLD and news_mult > 0:
            side = "long" if signal.value > 0 else "short"
            sizing = risk.position_size(symbol, equity, price, atr_val, side)
            entry_state = state_vector(signal.value, signal.confidence, atr_pct,
                                       0.0, 0.0, signal.value)
            action_name, mult = rl_agent.choose_size(entry_state)
            mult *= news_mult
            if mult > 0:
                sizing = risk.apply_rl_multiplier(sizing, mult, equity, symbol, price)
                if sizing.size > 0:
                    self.broker.submit_order(Order(
                        symbol=symbol, side=side, size=sizing.size,
                        stop_loss=sizing.stop_price,
                        metadata={"entry_state": entry_state.tolist(),
                                  "entry_action": action_name,
                                  "atr_pct": atr_pct,
                                  "ensemble_value": signal.value}), price, now)
                    result["action"] = f"opened_{side}"
                    result["size"] = sizing.size

        # periodic ML retrain
        maybe_retrain(archive, df, self.cycle_counts[key])
        self._persist(archive, ensemble, rl_agent)
        return result

    def _flatten(self, symbol, price, now, archive, ensemble, rl_agent, atr_pct):
        fill = self.broker.close_position(symbol, price, now)
        if fill is not None:
            self._learn_from_close(fill, archive, ensemble, rl_agent, atr_pct, now)

    def _learn_from_close(self, fill, archive: Archive, ensemble: Ensemble,
                          rl_agent: RLAgent, atr_pct: float, now) -> None:
        if fill is None:
            return
        pos = fill.__dict__.get("closed_position")
        pnl = fill.__dict__.get("pnl", 0.0)
        if pos is None:
            return
        entry_notional = pos.entry_price * pos.size
        pnl_pct = pnl / entry_notional if entry_notional else 0.0
        direction = 1.0 if pos.side == "long" else -1.0
        ensemble.update(direction * pnl_pct)
        if pos.entry_state is not None:
            import numpy as np

            rl_agent.update(np.asarray(pos.entry_state),
                            pos.entry_action or "size_1.0x",
                            trade_reward(pnl_pct, pos.atr_pct_at_entry or atr_pct),
                            None, done=True)
        archive.append_trade({
            "timestamp": now.isoformat(), "symbol": pos.symbol, "side": pos.side,
            "size": pos.size, "entry_price": pos.entry_price,
            "exit_price": fill.fill_price, "pnl": pnl, "pnl_pct": pnl_pct,
            "rl_action": pos.entry_action, "ensemble_signal": pos.ensemble_value,
        })

    def _persist(self, archive: Archive, ensemble: Ensemble, rl_agent: RLAgent):
        archive.save_weights_dict(ensemble.weights.to_dict())
        archive.save_rl_dict(rl_agent.to_dict())
