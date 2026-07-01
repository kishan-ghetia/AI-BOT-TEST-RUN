from trading_bot.risk.manager import RiskManager


def make_rm(**kw):
    defaults = dict(capital=10_000, risk_per_trade=0.01, max_dd_pct=0.20,
                    forex_leverage=30, crypto_leverage=1)
    defaults.update(kw)
    return RiskManager(**defaults)


def test_size_inverse_to_atr():
    rm = make_rm()
    small_atr = rm.position_size("BTC-USD", 10_000, 100.0, atr=1.0, side="long")
    big_atr = rm.position_size("BTC-USD", 10_000, 100.0, atr=5.0, side="long")
    assert small_atr.size > big_atr.size


def test_risk_amount_is_one_percent():
    rm = make_rm()
    sizing = rm.position_size("BTC-USD", 10_000, 100.0, atr=2.0, side="long")
    assert abs(sizing.risk_amount - 100.0) < 1e-6  # 1% of 10k


def test_stop_price_sides():
    rm = make_rm()
    long_s = rm.position_size("BTC-USD", 10_000, 100.0, atr=2.0, side="long")
    short_s = rm.position_size("BTC-USD", 10_000, 100.0, atr=2.0, side="short")
    assert long_s.stop_price < 100.0
    assert short_s.stop_price > 100.0


def test_leverage_margin_math():
    rm = make_rm()
    assert rm.margin_required(30_000, 30) == 1_000
    assert rm.margin_required(1_000, 1) == 1_000


def test_notional_capped_by_leverage():
    rm = make_rm(crypto_leverage=1)
    # tiny ATR would imply a huge position; must be capped at 1x equity
    sizing = rm.position_size("BTC-USD", 10_000, 100.0, atr=0.01, side="long")
    assert sizing.notional <= 10_000 * 1.0 + 1e-6


def test_forex_gets_more_notional_headroom():
    rm = make_rm()
    fx = rm.position_size("EURUSD=X", 10_000, 1.10, atr=0.0001, side="long")
    crypto = rm.position_size("BTC-USD", 10_000, 1.10, atr=0.0001, side="long")
    assert fx.notional > crypto.notional


def test_drawdown_breaker():
    rm = make_rm()
    assert not rm.check_drawdown_breaker([100, 95, 90, 85])       # -15%
    assert rm.check_drawdown_breaker([100, 90, 80])                # -20%
    assert rm.check_drawdown_breaker([100, 110, 121, 96.8])        # -20% from 121
    assert not rm.check_drawdown_breaker([100])


def test_rl_multiplier_scales_and_clamps():
    rm = make_rm(crypto_leverage=1)
    base = rm.position_size("BTC-USD", 10_000, 100.0, atr=2.0, side="long")
    scaled = rm.apply_rl_multiplier(base, 0.5, 10_000, "BTC-USD", 100.0)
    assert abs(scaled.size - base.size * 0.5) < 1e-9
    boosted = rm.apply_rl_multiplier(base, 100.0, 10_000, "BTC-USD", 100.0)
    assert boosted.notional <= 10_000 + 1e-6
