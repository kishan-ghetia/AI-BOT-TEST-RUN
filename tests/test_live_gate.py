from datetime import datetime, timedelta, timezone

import config
from trading_bot.archive.store import Archive
from trading_bot.live.gate import check_live_eligibility


def good_perf(now):
    started = (now - timedelta(weeks=6)).isoformat()
    return {"started_at": started, "last_run": now.isoformat(),
            "equity_curve": [], "sharpe": 1.8, "max_drawdown": 0.08,
            "trade_count": 40}


def test_blocked_without_flag(temp_store, monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", False)
    now = datetime.now(timezone.utc)
    a = Archive(temp_store, "daily", "TEST")
    a.save_performance(good_perf(now))
    gate = check_live_eligibility(a, now)
    assert not gate.allowed
    assert any("LIVE_TRADING_ENABLED" in r for r in gate.reasons)


def test_blocked_no_history(temp_store, monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    a = Archive(temp_store, "daily", "FRESH")
    gate = check_live_eligibility(a)
    assert not gate.allowed
    assert any("no paper trading history" in r for r in gate.reasons)


def test_blocked_too_young(temp_store, monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    now = datetime.now(timezone.utc)
    perf = good_perf(now)
    perf["started_at"] = (now - timedelta(weeks=1)).isoformat()
    a = Archive(temp_store, "daily", "YOUNG")
    a.save_performance(perf)
    gate = check_live_eligibility(a, now)
    assert not gate.allowed
    assert any("weeks continuous" in r for r in gate.reasons)


def test_blocked_low_sharpe(temp_store, monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    now = datetime.now(timezone.utc)
    perf = good_perf(now)
    perf["sharpe"] = 0.4
    a = Archive(temp_store, "daily", "LOWSHARPE")
    a.save_performance(perf)
    gate = check_live_eligibility(a, now)
    assert not gate.allowed
    assert any("Sharpe" in r for r in gate.reasons)


def test_blocked_high_drawdown(temp_store, monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    now = datetime.now(timezone.utc)
    perf = good_perf(now)
    perf["max_drawdown"] = 0.35
    a = Archive(temp_store, "daily", "BIGDD")
    a.save_performance(perf)
    gate = check_live_eligibility(a, now)
    assert not gate.allowed
    assert any("drawdown" in r for r in gate.reasons)


def test_allowed_when_all_criteria_met(temp_store, monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    now = datetime.now(timezone.utc)
    a = Archive(temp_store, "daily", "GOOD")
    a.save_performance(good_perf(now))
    gate = check_live_eligibility(a, now)
    assert gate.allowed
    assert gate.reasons == []
