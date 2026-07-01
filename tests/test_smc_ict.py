import pandas as pd

from trading_bot.strategies.smc_ict.fvg import find_fvgs, unfilled_fvgs
from trading_bot.strategies.smc_ict.liquidity import find_liquidity_pools, find_sweeps
from trading_bot.strategies.smc_ict.order_blocks import find_order_blocks
from trading_bot.strategies.smc_ict.sessions import session_weight
from trading_bot.strategies.smc_ict.structure import analyze_structure
from trading_bot.strategies.smc_ict.zones import premium_discount


def bars(rows):
    """rows: list of (open, high, low, close)."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1.0
    df.index = pd.date_range("2023-01-02", periods=len(df), freq="h")
    return df


def test_bullish_fvg_detected():
    # bar0 high=10; bar2 low=12 -> gap between 10 and 12 around bar1
    df = bars([
        (9, 10, 8, 9.5),
        (10, 13, 9.5, 12.5),
        (12.5, 14, 12, 13.5),
    ])
    gaps = find_fvgs(df)
    assert len(gaps) == 1
    g = gaps[0]
    assert g.kind == "bullish" and g.index == 1
    assert g.bottom == 10 and g.top == 12


def test_bearish_fvg_detected_and_fill_tracking():
    df = bars([
        (11, 12, 10, 10.5),
        (10.5, 10.6, 8, 8.2),
        (8.2, 9, 7.5, 8),      # high 9 < low 10 of bar0 => bearish gap 9..10
        (8, 12, 7.9, 11.5),    # rallies back through the gap -> filled
    ])
    gaps = find_fvgs(df)
    assert any(g.kind == "bearish" for g in gaps)
    assert unfilled_fvgs(df, gaps) == []


def test_order_block_bullish():
    # 25 quiet bars establish avg body, then a down candle followed by a huge up impulse
    quiet = [(100, 100.6, 99.4, 100.5) if i % 2 == 0 else (100.5, 101, 99.9, 100)
             for i in range(25)]
    down_candle = (100, 100.5, 98.5, 99)      # bearish
    impulse = (99, 106, 98.9, 105.5)          # big bullish body
    df = bars(quiet + [down_candle, impulse])
    blocks = find_order_blocks(df)
    assert any(b.kind == "bullish" and b.index == 25 for b in blocks)


def test_liquidity_pool_and_sweep():
    # two equal highs at 105, then a bar wicks to 105.5 but closes at 103 (sweep)
    rows = [(100, 101, 99, 100)] * 4
    rows += [(100, 105, 99.5, 101)]   # swing high 1 @105
    rows += [(101, 102, 100, 101)] * 4
    rows += [(101, 105.02, 100.5, 102)]  # swing high 2 ~105 (equal)
    rows += [(102, 103, 101, 102)] * 4
    rows += [(102, 105.5, 101.5, 103)]  # sweep: wick above, close below
    rows += [(103, 104, 102, 103)] * 3
    df = bars(rows)
    pools = find_liquidity_pools(df, tolerance_pct=0.002)
    assert any(p.kind == "highs" for p in pools)
    sweeps = find_sweeps(df, pools)
    assert any(s.kind == "high_sweep" for s in sweeps)


def test_structure_bull_trend_and_choch():
    # clean zigzag up: rally to 110, pull back to 106, rally through 110 (BOS)
    up = []
    for c in [100, 102, 104, 106, 108, 110]:          # leg 1 up, swing high 110
        up.append((c - 1, c + 0.5, c - 1.5, c))
    for c in [109, 108, 107, 106]:                    # pullback, swing low 106
        up.append((c + 1, c + 1.5, c - 0.5, c))
    for c in [108, 110, 112, 114, 116]:               # leg 2 breaks 110 -> BOS
        up.append((c - 1, c + 0.5, c - 1.5, c))
    df_up = bars(up)
    state = analyze_structure(df_up)
    assert state.trend == "bull"
    assert state.last_bos_index is not None

    # crash far below the last swing low => CHoCH, trend flips bear
    crash = up + [(116, 116, 85, 86)]
    state2 = analyze_structure(bars(crash))
    assert state2.trend == "bear"
    assert state2.last_choch_index == len(crash) - 1


def test_premium_discount():
    # rally to a 120 swing high, pull back to a 109 swing low, return to 118:
    # price sits in the upper part of the dealing range => premium
    rows = [(100 + i, 101 + i, 99 + i, 100 + i) for i in range(20)]  # high 120
    for c in [117, 115, 113, 111, 110]:                              # low 109
        rows.append((c + 1, c + 1, c - 1, c))
    for c in [112, 114, 116, 118]:
        rows.append((c - 1, c + 0.5, c - 1.5, c))
    zone, pos = premium_discount(bars(rows))
    assert zone == "premium"
    assert pos > 0.55


def test_session_weights():
    from datetime import datetime

    w_daily, name = session_weight(datetime(2023, 1, 2, 3, 0), "daily")
    assert w_daily == 1.0
    w_london, name = session_weight(datetime(2023, 1, 2, 8, 0), "m5")
    assert w_london == 1.0 and name == "london_open"
    w_dead, name = session_weight(datetime(2023, 1, 2, 22, 0), "m5")
    assert w_dead < 1.0
