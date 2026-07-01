from datetime import datetime, timedelta, timezone

from tests.conftest import make_ohlcv
from trading_bot.data.finnhub_client import FinnhubClient
from trading_bot.risk.news_filter import risk_multiplier
from trading_bot.strategies.news_strategy import NewsStrategy, surprise_score
from trading_bot.strategies.fibonacci import FibonacciStrategy, in_ote_zone


def test_finnhub_disabled_without_key():
    client = FinnhubClient(None)
    assert not client.enabled
    assert client.get_economic_calendar("2024-01-01", "2024-01-02") == []
    assert client.get_general_news() == []


def test_news_strategy_neutral_without_key(choppy):
    strat = NewsStrategy(FinnhubClient(None))
    sig = strat.generate_signal(choppy, {"symbol": "EURUSD=X", "timeframe": "daily"})
    assert sig.value == 0.0 and sig.confidence == 0.0


def test_surprise_score_signs_and_caps():
    assert surprise_score({"actual": 110, "estimate": 100}) > 0
    assert surprise_score({"actual": 90, "estimate": 100}) < 0
    assert surprise_score({"actual": 1000, "estimate": 1}) == 1.0
    assert surprise_score({"actual": None, "estimate": 100}) == 0.0


def test_news_filter_blackout_and_caution():
    now = datetime(2024, 6, 7, 12, 30, tzinfo=timezone.utc)
    nfp_soon = [{"impact": "high",
                 "time": (now + timedelta(minutes=10)).isoformat()}]
    assert risk_multiplier(nfp_soon, now) == 0.0
    nfp_later = [{"impact": "high",
                  "time": (now + timedelta(minutes=90)).isoformat()}]
    assert risk_multiplier(nfp_later, now) == 0.5
    nothing = [{"impact": "low",
                "time": (now + timedelta(minutes=5)).isoformat()}]
    assert risk_multiplier(nothing, now) == 1.0
    assert risk_multiplier([], now) == 1.0


def test_fib_strategy_produces_bounded_signal(trending_up):
    strat = FibonacciStrategy()
    sig = strat.generate_signal(trending_up, {"symbol": "T", "timeframe": "daily"})
    assert -1.0 <= sig.value <= 1.0
    assert 0.0 <= sig.confidence <= 1.0


def test_in_ote_zone_filter_api(trending_up):
    in_zone, side = in_ote_zone(trending_up)
    assert isinstance(in_zone, bool)
    assert side in ("long", "short", "none")
