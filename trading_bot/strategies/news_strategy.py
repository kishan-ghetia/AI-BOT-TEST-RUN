"""News strategy: trade surprises vs consensus from the Finnhub economic
calendar. Gracefully neutral when no FINNHUB_API_KEY is configured."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from trading_bot.data.finnhub_client import FinnhubClient
from trading_bot.strategies.base import Signal, Strategy

# currencies whose events matter per symbol prefix (rough mapping)
IMPACT_LEVELS = {"high": 1.0, "medium": 0.5, "low": 0.2, "3": 1.0, "2": 0.5, "1": 0.2}


def surprise_score(event: dict) -> float:
    """Signed, capped surprise: (actual - estimate) / |estimate|."""
    actual, estimate = event.get("actual"), event.get("estimate")
    if actual is None or estimate in (None, 0):
        return 0.0
    try:
        return max(-1.0, min(1.0, (float(actual) - float(estimate)) / abs(float(estimate))))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


class NewsStrategy(Strategy):
    name = "news"

    def __init__(self, client: FinnhubClient, window_hours: int = 12):
        self.client = client
        self.window_hours = window_hours

    def warmup_bars(self) -> int:
        return 1

    def generate_signal(self, df: pd.DataFrame, context: dict) -> Signal:
        if not self.client.enabled:
            return self.neutral(df, context, reason="no_api_key")

        now = datetime.now(timezone.utc)
        frm = (now - timedelta(hours=self.window_hours)).strftime("%Y-%m-%d")
        to = now.strftime("%Y-%m-%d")
        events = context.get("calendar_events")
        if events is None:
            events = self.client.get_economic_calendar(frm, to)

        # USD-centric heuristic: strong USD data is bearish for XXXUSD pairs
        # and crypto quoted in USD, bullish for USDXXX pairs.
        symbol = context.get("symbol", "")
        usd_is_quote = symbol.endswith("USD=X") or symbol.endswith("-USD")

        score = 0.0
        total_weight = 0.0
        for ev in events:
            if str(ev.get("country", "")).upper() not in ("US", "UNITED STATES"):
                continue
            impact = IMPACT_LEVELS.get(str(ev.get("impact", "")).lower(), 0.2)
            s = surprise_score(ev)
            if s == 0.0:
                continue
            direction = -1.0 if usd_is_quote else 1.0
            score += direction * s * impact
            total_weight += impact

        if total_weight == 0:
            return self.neutral(df, context, reason="no_surprises")
        value = score / total_weight
        return Signal(
            symbol=symbol,
            timeframe=context.get("timeframe", "?"),
            timestamp=df.index[-1] if len(df) else now,
            strategy_name=self.name,
            value=value,
            confidence=min(1.0, total_weight),
            metadata={"events_scored": int(total_weight > 0)},
        ).clamped()
