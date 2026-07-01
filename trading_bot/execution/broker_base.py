"""Broker abstraction shared by paper, backtest, and (future) live adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class BrokerNotConfiguredError(RuntimeError):
    """Raised when a live broker adapter is constructed without credentials."""


@dataclass
class Order:
    symbol: str
    side: str            # "long" | "short" (target direction)
    size: float
    order_type: str = "market"
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Fill:
    symbol: str
    side: str
    size: float
    fill_price: float
    fee: float
    timestamp: Any = None


@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: Any = None
    bars_held: int = 0
    entry_state: list | None = None   # RL state snapshot at entry
    entry_action: str | None = None   # RL sizing action taken
    atr_pct_at_entry: float = 0.0
    ensemble_value: float = 0.0

    def unrealized_pnl(self, price: float) -> float:
        direction = 1.0 if self.side == "long" else -1.0
        return direction * (price - self.entry_price) * self.size

    def unrealized_pnl_pct(self, price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        direction = 1.0 if self.side == "long" else -1.0
        return direction * (price - self.entry_price) / self.entry_price


class Broker(ABC):
    @abstractmethod
    def get_balance(self) -> float: ...

    @abstractmethod
    def get_equity(self, prices: dict[str, float]) -> float: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def submit_order(self, order: Order, price: float, timestamp=None) -> Fill: ...

    @abstractmethod
    def close_position(self, symbol: str, price: float, timestamp=None) -> Fill | None: ...
