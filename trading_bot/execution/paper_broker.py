"""Paper broker: in-memory simulated ledger with slippage + fees.

Shared by the backtest engine and `paper` mode so fills behave identically.
"""
from __future__ import annotations

import config
from trading_bot.execution.broker_base import Broker, Fill, Order, Position


class PaperBroker(Broker):
    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL,
                 slippage_bps: float = config.SLIPPAGE_BPS,
                 fee_bps: float = config.FEE_BPS):
        self.cash = initial_capital
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.fills: list[Fill] = []

    # --- account state ---
    def get_balance(self) -> float:
        return self.cash

    def get_equity(self, prices: dict[str, float]) -> float:
        equity = self.cash
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.entry_price)
            equity += pos.unrealized_pnl(price)
        return equity

    def get_positions(self) -> list[Position]:
        return list(self.positions.values())

    # --- execution ---
    def _slip(self, price: float, side: str) -> float:
        slip = price * self.slippage_bps / 10_000.0
        return price + slip if side == "long" else price - slip

    def submit_order(self, order: Order, price: float, timestamp=None) -> Fill:
        if order.symbol in self.positions:
            self.close_position(order.symbol, price, timestamp)
        fill_price = self._slip(price, order.side)
        fee = abs(order.size * fill_price) * self.fee_bps / 10_000.0
        self.cash -= fee
        pos = Position(
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            entry_price=fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            opened_at=timestamp,
            entry_state=order.metadata.get("entry_state"),
            entry_action=order.metadata.get("entry_action"),
            atr_pct_at_entry=order.metadata.get("atr_pct", 0.0),
            ensemble_value=order.metadata.get("ensemble_value", 0.0),
        )
        self.positions[order.symbol] = pos
        fill = Fill(order.symbol, order.side, order.size, fill_price, fee, timestamp)
        self.fills.append(fill)
        return fill

    def close_position(self, symbol: str, price: float, timestamp=None) -> Fill | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        exit_side = "short" if pos.side == "long" else "long"
        fill_price = self._slip(price, exit_side)
        fee = abs(pos.size * fill_price) * self.fee_bps / 10_000.0
        pnl = pos.unrealized_pnl(fill_price) - fee
        self.cash += pnl
        self.realized_pnl += pnl
        fill = Fill(symbol, exit_side, pos.size, fill_price, fee, timestamp)
        fill_pnl = pnl  # convenience for callers
        fill.__dict__["pnl"] = fill_pnl
        fill.__dict__["closed_position"] = pos
        self.fills.append(fill)
        return fill

    def check_stops(self, symbol: str, bar_high: float, bar_low: float,
                    timestamp=None) -> Fill | None:
        """Trigger stop-loss/take-profit intrabar. Called once per new bar."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        if pos.side == "long":
            if pos.stop_loss is not None and bar_low <= pos.stop_loss:
                return self.close_position(symbol, pos.stop_loss, timestamp)
            if pos.take_profit is not None and bar_high >= pos.take_profit:
                return self.close_position(symbol, pos.take_profit, timestamp)
        else:
            if pos.stop_loss is not None and bar_high >= pos.stop_loss:
                return self.close_position(symbol, pos.stop_loss, timestamp)
            if pos.take_profit is not None and bar_low <= pos.take_profit:
                return self.close_position(symbol, pos.take_profit, timestamp)
        return None
