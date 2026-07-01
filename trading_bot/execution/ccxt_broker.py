"""Crypto broker adapter via ccxt. Structurally complete; requires
EXCHANGE_API_KEY/SECRET in .env to construct. Testnet supported."""
from __future__ import annotations

import logging

import config
from trading_bot.execution.broker_base import (
    Broker, BrokerNotConfiguredError, Fill, Order, Position,
)

logger = logging.getLogger(__name__)


def yf_to_ccxt(symbol: str) -> str:
    """BTC-USD -> BTC/USDT (ccxt market naming)."""
    base = symbol.replace("-USD", "")
    return f"{base}/USDT"


class CcxtBroker(Broker):
    def __init__(self, exchange_id: str = config.EXCHANGE_ID,
                 api_key: str = config.EXCHANGE_API_KEY,
                 api_secret: str = config.EXCHANGE_API_SECRET,
                 testnet: bool = config.EXCHANGE_TESTNET):
        if not api_key or not api_secret:
            raise BrokerNotConfiguredError(
                "EXCHANGE_API_KEY / EXCHANGE_API_SECRET not set in .env - "
                "CcxtBroker unavailable, use PaperBroker.")
        import ccxt

        cls = getattr(ccxt, exchange_id)
        self.exchange = cls({"apiKey": api_key, "secret": api_secret,
                             "enableRateLimit": True})
        if testnet and self.exchange.has.get("sandbox"):
            self.exchange.set_sandbox_mode(True)

    def get_balance(self) -> float:
        bal = self.exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0.0))

    def get_equity(self, prices: dict[str, float]) -> float:
        bal = self.exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("total", 0.0))

    def get_positions(self) -> list[Position]:
        positions = []
        bal = self.exchange.fetch_balance()
        for currency, amounts in bal.items():
            if currency in ("USDT", "info", "free", "used", "total"):
                continue
            total = amounts.get("total") if isinstance(amounts, dict) else None
            if total:
                positions.append(Position(
                    symbol=f"{currency}-USD", side="long",
                    size=float(total), entry_price=0.0))
        return positions

    def submit_order(self, order: Order, price: float, timestamp=None) -> Fill:
        market = yf_to_ccxt(order.symbol)
        side = "buy" if order.side == "long" else "sell"
        result = self.exchange.create_market_order(market, side, order.size)
        fill_price = float(result.get("average") or result.get("price") or price)
        fee = float((result.get("fee") or {}).get("cost") or 0.0)
        return Fill(order.symbol, order.side, order.size, fill_price, fee, timestamp)

    def close_position(self, symbol: str, price: float, timestamp=None) -> Fill | None:
        market = yf_to_ccxt(symbol)
        base = market.split("/")[0]
        bal = self.exchange.fetch_balance()
        amount = float(bal.get(base, {}).get("free", 0.0))
        if amount <= 0:
            return None
        result = self.exchange.create_market_order(market, "sell", amount)
        fill_price = float(result.get("average") or result.get("price") or price)
        fee = float((result.get("fee") or {}).get("cost") or 0.0)
        return Fill(symbol, "short", amount, fill_price, fee, timestamp)
