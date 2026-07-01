"""Shared fixtures: synthetic OHLCV data, temp data_store, and a hard
guard that fails any test that tries to hit the network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("Network call attempted in tests")

    import requests

    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)
    monkeypatch.setattr(requests.Session, "get", _blocked)
    monkeypatch.setattr(requests.Session, "post", _blocked)
    try:
        import yfinance

        monkeypatch.setattr(yfinance, "download", _blocked)
    except ImportError:
        pass


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "DATA_STORE_DIR", tmp_path)
    return tmp_path


def make_ohlcv(n: int = 300, trend: float = 0.0, vol: float = 0.01,
               seed: int = 42, start_price: float = 100.0,
               freq: str = "D") -> pd.DataFrame:
    """Deterministic synthetic OHLCV series."""
    rng = np.random.default_rng(seed)
    rets = trend + vol * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(vol * rng.standard_normal(n)) * 0.5)
    low = close * (1 - np.abs(vol * rng.standard_normal(n)) * 0.5)
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum.reduce([high, close, open_])
    low = np.minimum.reduce([low, close, open_])
    volume = rng.uniform(1e5, 1e6, n)
    idx = pd.date_range("2023-01-02", periods=n, freq=freq)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


@pytest.fixture
def trending_up():
    return make_ohlcv(300, trend=0.003, vol=0.01)


@pytest.fixture
def trending_down():
    return make_ohlcv(300, trend=-0.003, vol=0.01)


@pytest.fixture
def choppy():
    return make_ohlcv(300, trend=0.0, vol=0.015)
