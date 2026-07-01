"""Event-driven backtest engine.

Walks bars chronologically; at each bar the ensemble sees only history up to
that bar, the risk manager sizes the trade, the RL agent scales it, and the
PaperBroker simulates fills with slippage/fees. Adaptive weights and the RL
agent learn during the run (that's the point - the backtest doubles as a
training gym for the self-learning components).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

import config
from trading_bot.execution.broker_base import Order
from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.learning.ensemble import Ensemble
from trading_bot.learning.rl_agent import RLAgent, state_vector, trade_reward
from trading_bot.risk.manager import RiskManager
from trading_bot.strategies import indicators as ta
from trading_bot.backtest.metrics import compute_metrics

logger = logging.getLogger(__name__)

ENTRY_THRESHOLD = 0.15  # |ensemble value| below this = no trade


@dataclass
class BacktestResult:
    equity_curve: list[float] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    final_weights: dict = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, ensemble: Ensemble, symbol: str,
                 timeframe: str, rl_agent: RLAgent | None = None,
                 risk: RiskManager | None = None,
                 initial_capital: float = config.INITIAL_CAPITAL,
                 warmup: int = 150, rl_greedy: bool = False) -> BacktestResult:
    risk = risk or RiskManager(capital=initial_capital)
    broker = PaperBroker(initial_capital=initial_capital)
    result = BacktestResult()
    atr_series = ta.atr(df)
    halted = False

    for i in range(warmup, len(df)):
        window = df.iloc[: i + 1]
        bar = df.iloc[i]
        ts = df.index[i]
        price = float(bar["close"])
        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else price * 0.01
        atr_pct = atr / price if price > 0 else 0.01
        prices = {symbol: price}
        equity = broker.get_equity(prices)
        result.equity_curve.append(equity)

        # circuit breaker: stop opening new positions, flatten existing
        if not halted and risk.check_drawdown_breaker(result.equity_curve):
            logger.warning("Drawdown breaker tripped at bar %s", ts)
            halted = True
            fill = broker.close_position(symbol, price, ts)
            _record_close(result, ensemble, rl_agent, fill, ts, symbol, atr_pct)
        if halted:
            continue

        # stop-loss / take-profit intrabar checks
        fill = broker.check_stops(symbol, float(bar["high"]), float(bar["low"]), ts)
        if fill is not None:
            _record_close(result, ensemble, rl_agent, fill, ts, symbol, atr_pct)

        signal = ensemble.combine(window, {"symbol": symbol, "timeframe": timeframe})
        pos = broker.positions.get(symbol)

        if pos is not None:
            pos.bars_held += 1
            # RL exit decision
            if rl_agent is not None:
                st = state_vector(signal.value, signal.confidence, atr_pct,
                                  pos.unrealized_pnl_pct(price), pos.bars_held,
                                  signal.value)
                if rl_agent.choose_exit(st, greedy=rl_greedy) == "close":
                    fill = broker.close_position(symbol, price, ts)
                    _record_close(result, ensemble, rl_agent, fill, ts, symbol, atr_pct)
                    pos = None
            # ensemble flip = close
            if pos is not None:
                flipped = (pos.side == "long" and signal.value < -ENTRY_THRESHOLD) or (
                    pos.side == "short" and signal.value > ENTRY_THRESHOLD)
                if flipped:
                    fill = broker.close_position(symbol, price, ts)
                    _record_close(result, ensemble, rl_agent, fill, ts, symbol, atr_pct)
                    pos = None

        # entries
        if pos is None and abs(signal.value) >= ENTRY_THRESHOLD:
            side = "long" if signal.value > 0 else "short"
            sizing = risk.position_size(symbol, equity, price, atr, side)
            multiplier = 1.0
            action = "size_1.0x"
            entry_state = None
            if rl_agent is not None:
                entry_state = state_vector(signal.value, signal.confidence,
                                           atr_pct, 0.0, 0.0, signal.value)
                action, multiplier = rl_agent.choose_size(entry_state, greedy=rl_greedy)
            if multiplier > 0:
                sizing = risk.apply_rl_multiplier(sizing, multiplier, equity,
                                                  symbol, price)
                if sizing.size > 0:
                    broker.submit_order(Order(
                        symbol=symbol, side=side, size=sizing.size,
                        stop_loss=sizing.stop_price,
                        metadata={
                            "entry_state": None if entry_state is None else entry_state.tolist(),
                            "entry_action": action,
                            "atr_pct": atr_pct,
                            "ensemble_value": signal.value,
                        }), price, ts)

    # close any open position at the end
    if broker.positions.get(symbol) is not None:
        last_price = float(df["close"].iloc[-1])
        last_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else last_price * 0.01
        fill = broker.close_position(symbol, last_price, df.index[-1])
        _record_close(result, ensemble, rl_agent, fill, df.index[-1], symbol,
                      last_atr / last_price)

    result.equity_curve.append(broker.get_equity({symbol: float(df["close"].iloc[-1])}))
    result.metrics = compute_metrics(result.equity_curve, timeframe, result.trades)
    result.final_weights = dict(ensemble.weights.weights)
    return result


def _record_close(result: BacktestResult, ensemble: Ensemble,
                  rl_agent: RLAgent | None, fill, ts, symbol: str,
                  atr_pct: float) -> None:
    """Log a closed trade and feed rewards to the learners."""
    if fill is None:
        return
    pos = fill.__dict__.get("closed_position")
    pnl = fill.__dict__.get("pnl", 0.0)
    if pos is None:
        return
    entry_notional = pos.entry_price * pos.size
    pnl_pct = pnl / entry_notional if entry_notional else 0.0

    result.trades.append({
        "timestamp": ts, "symbol": symbol, "side": pos.side, "size": pos.size,
        "entry_price": pos.entry_price, "exit_price": fill.fill_price,
        "pnl": pnl, "pnl_pct": pnl_pct, "rl_action": pos.entry_action,
        "ensemble_signal": pos.ensemble_value,
    })

    # adaptive weights learn from the realized, direction-signed return
    direction = 1.0 if pos.side == "long" else -1.0
    ensemble.update(direction * pnl_pct)

    # RL learns from risk-adjusted reward on the entry action
    if rl_agent is not None and pos.entry_state is not None:
        import numpy as np

        reward = trade_reward(pnl_pct, pos.atr_pct_at_entry or atr_pct)
        rl_agent.update(np.asarray(pos.entry_state), pos.entry_action or "size_1.0x",
                        reward, None, done=True)
