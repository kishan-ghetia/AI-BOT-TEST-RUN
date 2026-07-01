"""Per-(timeframe, symbol) persisted state: weights, ML model, RL agent,
performance history, trade log. Everything the self-learning loop needs to
survive restarts lives here.

Layout: data_store/<timeframe>/<SYMBOL>/{weights.json, model.pkl,
model_meta.json, rl_agent.json, performance.json, trades.csv}
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

TRADE_COLUMNS = [
    "timestamp", "symbol", "side", "size", "entry_price", "exit_price",
    "pnl", "pnl_pct", "rl_action", "ensemble_signal",
]


class Archive:
    def __init__(self, base_dir: Path, timeframe: str, symbol: str):
        self.dir = Path(base_dir) / timeframe / symbol
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeframe = timeframe
        self.symbol = symbol

    # --- generic json helpers ---
    def _read_json(self, name: str) -> dict | None:
        path = self.dir / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt archive file %s: %s", path, exc)
            return None

    def _write_json(self, name: str, data: dict) -> None:
        tmp = self.dir / f".{name}.tmp"
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(self.dir / name)

    # --- ensemble weights ---
    def load_weights_dict(self) -> dict | None:
        return self._read_json("weights.json")

    def save_weights_dict(self, data: dict) -> None:
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._write_json("weights.json", data)

    # --- ML model ---
    def load_model(self) -> Any | None:
        path = self.dir / "model.pkl"
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception as exc:
            logger.warning("Failed loading model %s: %s", path, exc)
            return None

    def save_model(self, model: Any) -> None:
        joblib.dump(model, self.dir / "model.pkl")

    # --- RL agent ---
    def load_rl_dict(self) -> dict | None:
        return self._read_json("rl_agent.json")

    def save_rl_dict(self, data: dict) -> None:
        self._write_json("rl_agent.json", data)

    # --- performance history (feeds the live gate + dashboard) ---
    def load_performance(self) -> dict:
        return self._read_json("performance.json") or {
            "started_at": None,
            "last_run": None,
            "equity_curve": [],
            "sharpe": None,
            "max_drawdown": None,
            "trade_count": 0,
        }

    def update_performance(self, equity: float, ts: datetime,
                           cadence_seconds: float | None = None) -> dict:
        """Append an equity point. Resets started_at if the run gap exceeds
        2x cadence (a paper bot left off must re-earn its continuity)."""
        perf = self.load_performance()
        now_iso = ts.isoformat()
        if perf["started_at"] is None:
            perf["started_at"] = now_iso
        elif cadence_seconds and perf["last_run"]:
            last = datetime.fromisoformat(perf["last_run"])
            if (ts - last).total_seconds() > 2 * cadence_seconds:
                logger.info("Run gap too large for %s/%s: resetting continuity",
                            self.timeframe, self.symbol)
                perf["started_at"] = now_iso
                perf["equity_curve"] = []
        perf["last_run"] = now_iso
        perf["equity_curve"].append([now_iso, equity])

        curve = [pt[1] for pt in perf["equity_curve"]]
        perf["sharpe"] = _rolling_sharpe(curve)
        perf["max_drawdown"] = _max_drawdown(curve)
        self._write_json("performance.json", perf)
        return perf

    def save_performance(self, perf: dict) -> None:
        self._write_json("performance.json", perf)

    # --- trade log ---
    def append_trade(self, trade: dict) -> None:
        path = self.dir / "trades.csv"
        new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_COLUMNS, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(trade)
        perf = self.load_performance()
        perf["trade_count"] = perf.get("trade_count", 0) + 1
        self._write_json("performance.json", perf)


def _rolling_sharpe(curve: list[float]) -> float | None:
    """Annualization-free Sharpe on equity-curve step returns."""
    import numpy as np

    if len(curve) < 3:
        return None
    rets = np.diff(curve) / np.array(curve[:-1])
    std = rets.std()
    if std == 0:
        return 0.0
    # scale by sqrt(steps/year-ish) is timeframe-dependent; the gate compares
    # against a threshold on this same statistic, so keep it consistent:
    return float(rets.mean() / std * np.sqrt(252))


def _max_drawdown(curve: list[float]) -> float | None:
    import numpy as np

    if len(curve) < 2:
        return None
    arr = np.asarray(curve, dtype=float)
    peaks = np.maximum.accumulate(arr)
    dd = (peaks - arr) / peaks
    return float(dd.max())
