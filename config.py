"""Central configuration for the trading bot.

All values can be overridden via environment variables (loaded from .env).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_STORE_DIR = Path(os.getenv("DATA_STORE_DIR", ROOT_DIR / "data_store"))

# --- Capital & risk ---
INITIAL_CAPITAL = _env_float("INITIAL_CAPITAL", 10_000.0)
RISK_PER_TRADE = _env_float("RISK_PER_TRADE", 0.01)
MAX_DRAWDOWN_PCT = _env_float("MAX_DRAWDOWN_PCT", 0.20)
FOREX_LEVERAGE = _env_float("FOREX_LEVERAGE", 30.0)
CRYPTO_LEVERAGE = _env_float("CRYPTO_LEVERAGE", 1.0)

# --- Execution simulation ---
SLIPPAGE_BPS = _env_float("SLIPPAGE_BPS", 2.0)
FEE_BPS = _env_float("FEE_BPS", 5.0)

# --- Live trading hard gate ---
LIVE_TRADING_ENABLED = _env_bool("LIVE_TRADING_ENABLED", False)
GATE_MIN_PAPER_WEEKS = _env_float("GATE_MIN_PAPER_WEEKS", 4.0)
GATE_MIN_SHARPE = _env_float("GATE_MIN_SHARPE", 1.0)
GATE_MAX_DRAWDOWN = _env_float("GATE_MAX_DRAWDOWN", 0.20)

# --- API keys (optional; modules no-op or refuse gracefully without them) ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")
EXCHANGE_TESTNET = _env_bool("EXCHANGE_TESTNET", True)
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_PRACTICE = _env_bool("OANDA_PRACTICE", True)

# --- Data ---
# When true, fall back to synthetic data if real market data is unreachable
# (offline environments / CI). Paper results on synthetic data do NOT count
# toward anything real - the scheduler logs loudly when this happens.
SYNTHETIC_FALLBACK = _env_bool("SYNTHETIC_FALLBACK", False)

# --- Timeframes ---
TIMEFRAMES = ("daily", "hourly", "m5")
