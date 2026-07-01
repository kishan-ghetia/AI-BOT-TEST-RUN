import numpy as np
import pandas as pd

from trading_bot.strategies import indicators as ta
from tests.conftest import make_ohlcv


def test_sma_hand_computed():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ta.sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == 2.0
    assert out.iloc[4] == 4.0


def test_ema_converges_to_constant():
    s = pd.Series([10.0] * 50)
    assert abs(ta.ema(s, 10).iloc[-1] - 10.0) < 1e-9


def test_rsi_extremes():
    up = pd.Series(np.linspace(1, 100, 50))
    down = pd.Series(np.linspace(100, 1, 50))
    assert ta.rsi(up).iloc[-1] > 90
    assert ta.rsi(down).iloc[-1] < 10


def test_macd_positive_in_uptrend():
    up = pd.Series(np.exp(np.linspace(0, 1, 100)))
    macd_line, _, hist = ta.macd(up)
    assert macd_line.iloc[-1] > 0


def test_bollinger_contains_price_mostly():
    df = make_ohlcv(200)
    upper, mid, lower = ta.bollinger(df["close"])
    valid = ~upper.isna()
    inside = ((df["close"] <= upper) & (df["close"] >= lower))[valid]
    assert inside.mean() > 0.8


def test_atr_positive_and_scales_with_vol():
    calm = make_ohlcv(200, vol=0.005, seed=1)
    wild = make_ohlcv(200, vol=0.03, seed=1)
    atr_calm = ta.atr(calm).iloc[-1]
    atr_wild = ta.atr(wild).iloc[-1]
    assert atr_calm > 0
    assert atr_wild > atr_calm


def test_swing_points_detects_obvious_peak():
    # construct a series with one clear peak in the middle
    prices = [10, 11, 12, 15, 12, 11, 10, 9, 8, 9, 10, 11]
    df = pd.DataFrame({
        "open": prices, "close": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "volume": [1] * len(prices),
    }, index=pd.date_range("2023-01-01", periods=len(prices)))
    sh, sl = ta.swing_points(df, lookback=3)
    assert sh.iloc[3]  # the peak at index 3
    assert sl.iloc[8]  # the trough at index 8
