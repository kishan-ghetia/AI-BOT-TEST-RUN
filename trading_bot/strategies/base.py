"""Core strategy interface: every signal source implements Strategy -> Signal."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Signal:
    symbol: str
    timeframe: str
    timestamp: Any
    strategy_name: str
    value: float  # [-1, 1]: -1 strong short, +1 strong long
    confidence: float  # [0, 1]
    metadata: dict = field(default_factory=dict)

    def clamped(self) -> "Signal":
        self.value = max(-1.0, min(1.0, self.value))
        self.confidence = max(0.0, min(1.0, self.confidence))
        return self


class Strategy(ABC):
    """A signal source. `context` carries symbol, timeframe, and shared
    resources (archive handle, news snapshot, fib zones...)."""

    name: str = "base"

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, context: dict) -> Signal:
        ...

    def warmup_bars(self) -> int:
        return 50

    def neutral(self, df: pd.DataFrame, context: dict, **meta) -> Signal:
        ts = df.index[-1] if len(df) else None
        return Signal(
            symbol=context.get("symbol", "?"),
            timeframe=context.get("timeframe", "?"),
            timestamp=ts,
            strategy_name=self.name,
            value=0.0,
            confidence=0.0,
            metadata=meta,
        )
