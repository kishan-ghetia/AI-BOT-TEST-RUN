"""Thin Finnhub REST wrapper. Gracefully no-ops when no API key is configured."""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://finnhub.io/api/v1"


class FinnhubClient:
    def __init__(self, api_key: str | None):
        self.api_key = api_key or ""
        self.enabled = bool(self.api_key)

    def _get(self, path: str, params: dict) -> dict | list:
        if not self.enabled:
            return {}
        params = dict(params, token=self.api_key)
        try:
            resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Finnhub request failed (%s): %s", path, exc)
            return {}

    def get_economic_calendar(self, frm: str, to: str) -> list[dict]:
        """Scheduled macro events with actual/estimate/prev/impact. [] if disabled."""
        data = self._get("/calendar/economic", {"from": frm, "to": to})
        return data.get("economicCalendar", []) if isinstance(data, dict) else []

    def get_general_news(self, category: str = "forex") -> list[dict]:
        """Latest market news headlines. [] if disabled."""
        data = self._get("/news", {"category": category})
        return data if isinstance(data, list) else []
