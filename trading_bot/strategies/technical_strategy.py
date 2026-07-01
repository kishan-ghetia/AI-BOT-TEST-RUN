"""Rule-based technical strategy: EMA crossover + RSI + MACD + Bollinger vote."""
from __future__ import annotations

import pandas as pd

from trading_bot.strategies import indicators as ta
from trading_bot.strategies.base import Signal, Strategy


class TechnicalStrategy(Strategy):
    name = "technical"

    def __init__(self, fast: int = 10, slow: int = 50, rsi_window: int = 14):
        self.fast = fast
        self.slow = slow
        self.rsi_window = rsi_window

    def warmup_bars(self) -> int:
        return self.slow + 10

    def generate_signal(self, df: pd.DataFrame, context: dict) -> Signal:
        if len(df) < self.warmup_bars():
            return self.neutral(df, context, reason="warmup")
        close = df["close"]
        votes: list[float] = []

        # EMA trend
        fast_ema = ta.ema(close, self.fast).iloc[-1]
        slow_ema = ta.ema(close, self.slow).iloc[-1]
        votes.append(1.0 if fast_ema > slow_ema else -1.0)

        # RSI mean-reversion at extremes, trend-follow otherwise
        r = ta.rsi(close, self.rsi_window).iloc[-1]
        if r < 30:
            votes.append(1.0)
        elif r > 70:
            votes.append(-1.0)
        else:
            votes.append((r - 50) / 50.0)

        # MACD histogram direction
        _, _, hist = ta.macd(close)
        votes.append(1.0 if hist.iloc[-1] > 0 else -1.0)

        # Bollinger position: fade the bands
        upper, _, lower = ta.bollinger(close)
        price = close.iloc[-1]
        if price < lower.iloc[-1]:
            votes.append(1.0)
        elif price > upper.iloc[-1]:
            votes.append(-1.0)
        else:
            votes.append(0.0)

        value = sum(votes) / len(votes)
        # confidence rises with vote agreement
        agreement = abs(value)
        return Signal(
            symbol=context.get("symbol", "?"),
            timeframe=context.get("timeframe", "?"),
            timestamp=df.index[-1],
            strategy_name=self.name,
            value=value,
            confidence=0.3 + 0.7 * agreement,
            metadata={"votes": votes, "rsi": float(r)},
        ).clamped()
