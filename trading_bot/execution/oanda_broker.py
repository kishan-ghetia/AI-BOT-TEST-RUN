"""Forex broker adapter via OANDA v20 REST API. Structurally complete;
requires OANDA_API_KEY + OANDA_ACCOUNT_ID in .env to construct."""
from __future__ import annotations

import logging

import requests

import config
from trading_bot.execution.broker_base import (
    Broker, BrokerNotConfiguredError, Fill, Order, Position,
)

logger = logging.getLogger(__name__)

PRACTICE_URL = "https://api-fxpractice.oanda.com"
LIVE_URL = "https://api-fxtrade.oanda.com"


def yf_to_oanda(symbol: str) -> str:
    """EURUSD=X -> EUR_USD."""
    pair = symbol.replace("=X", "")
    return f"{pair[:3]}_{pair[3:]}"


class OandaBroker(Broker):
    def __init__(self, api_key: str = config.OANDA_API_KEY,
                 account_id: str = config.OANDA_ACCOUNT_ID,
                 practice: bool = config.OANDA_PRACTICE):
        if not api_key or not account_id:
            raise BrokerNotConfiguredError(
                "OANDA_API_KEY / OANDA_ACCOUNT_ID not set in .env - "
                "OandaBroker unavailable, use PaperBroker.")
        self.base_url = PRACTICE_URL if practice else LIVE_URL
        self.account_id = account_id
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _get(self, path: str) -> dict:
        resp = self.session.get(f"{self.base_url}{path}", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> dict:
        resp = self.session.post(f"{self.base_url}{path}", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_balance(self) -> float:
        data = self._get(f"/v3/accounts/{self.account_id}/summary")
        return float(data["account"]["balance"])

    def get_equity(self, prices: dict[str, float]) -> float:
        data = self._get(f"/v3/accounts/{self.account_id}/summary")
        return float(data["account"]["NAV"])

    def get_positions(self) -> list[Position]:
        data = self._get(f"/v3/accounts/{self.account_id}/openPositions")
        out = []
        for p in data.get("positions", []):
            long_units = float(p["long"]["units"])
            short_units = float(p["short"]["units"])
            units = long_units if long_units else short_units
            side = "long" if units > 0 else "short"
            leg = p["long"] if units > 0 else p["short"]
            out.append(Position(
                symbol=p["instrument"].replace("_", "") + "=X",
                side=side,
                size=abs(units),
                entry_price=float(leg.get("averagePrice", 0.0)),
            ))
        return out

    def submit_order(self, order: Order, price: float, timestamp=None) -> Fill:
        units = int(order.size) if order.side == "long" else -int(order.size)
        payload = {"order": {
            "instrument": yf_to_oanda(order.symbol),
            "units": str(units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }}
        if order.stop_loss:
            payload["order"]["stopLossOnFill"] = {"price": f"{order.stop_loss:.5f}"}
        if order.take_profit:
            payload["order"]["takeProfitOnFill"] = {"price": f"{order.take_profit:.5f}"}
        data = self._post(f"/v3/accounts/{self.account_id}/orders", payload)
        tx = data.get("orderFillTransaction", {})
        fill_price = float(tx.get("price", price))
        return Fill(order.symbol, order.side, abs(units), fill_price, 0.0, timestamp)

    def close_position(self, symbol: str, price: float, timestamp=None) -> Fill | None:
        instrument = yf_to_oanda(symbol)
        payload = {"longUnits": "ALL", "shortUnits": "ALL"}
        try:
            data = self._post(
                f"/v3/accounts/{self.account_id}/positions/{instrument}/close",
                payload)
        except requests.HTTPError:
            return None
        tx = data.get("longOrderFillTransaction") or data.get("shortOrderFillTransaction") or {}
        fill_price = float(tx.get("price", price))
        units = abs(float(tx.get("units", 0)))
        return Fill(symbol, "close", units, fill_price, 0.0, timestamp)
