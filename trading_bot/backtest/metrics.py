"""Performance metrics for backtests and paper runs."""
from __future__ import annotations

import numpy as np

PERIODS_PER_YEAR = {"daily": 252, "hourly": 252 * 24, "m5": 252 * 24 * 12}


def compute_metrics(equity_curve: list[float], timeframe: str = "daily",
                    trades: list[dict] | None = None) -> dict:
    arr = np.asarray(equity_curve, dtype=float)
    out: dict = {"bars": len(arr)}
    if len(arr) < 2:
        return {**out, "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "win_rate": None, "trade_count": 0}

    rets = np.diff(arr) / arr[:-1]
    periods = PERIODS_PER_YEAR.get(timeframe, 252)
    total_return = arr[-1] / arr[0] - 1.0
    years = len(arr) / periods
    cagr = (arr[-1] / arr[0]) ** (1 / years) - 1.0 if years > 0 and arr[0] > 0 else 0.0
    std = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(periods)) if std > 0 else 0.0
    peaks = np.maximum.accumulate(arr)
    max_dd = float(((peaks - arr) / peaks).max())

    out.update(total_return=float(total_return), cagr=float(cagr),
               sharpe=sharpe, max_drawdown=max_dd)

    if trades:
        pnls = [t.get("pnl", 0.0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        out["trade_count"] = len(pnls)
        out["win_rate"] = wins / len(pnls) if pnls else None
        out["avg_pnl"] = float(np.mean(pnls)) if pnls else 0.0
    else:
        out["trade_count"] = 0
        out["win_rate"] = None
    return out


def format_metrics(m: dict) -> str:
    lines = [
        f"  Bars:          {m.get('bars', 0)}",
        f"  Total return:  {m.get('total_return', 0) * 100:+.2f}%",
        f"  CAGR:          {m.get('cagr', 0) * 100:+.2f}%",
        f"  Sharpe:        {m.get('sharpe', 0):.2f}",
        f"  Max drawdown:  {m.get('max_drawdown', 0) * 100:.2f}%",
        f"  Trades:        {m.get('trade_count', 0)}",
    ]
    if m.get("win_rate") is not None:
        lines.append(f"  Win rate:      {m['win_rate'] * 100:.1f}%")
    return "\n".join(lines)
