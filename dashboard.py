"""Streamlit monitoring dashboard: equity curves, trades, ensemble weights,
and evolution population - read straight from data_store/.

Run: streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="AI Trading Bot", layout="wide")
st.title("AI Trading Bot - Monitor")

base = Path(config.DATA_STORE_DIR)
timeframes = [d.name for d in base.iterdir() if d.is_dir()
              and d.name in config.TIMEFRAMES] if base.exists() else []

if not timeframes:
    st.info("No paper-trading data yet. Run: python main.py paper --symbols BTC-USD")
    st.stop()

tf = st.sidebar.selectbox("Timeframe", timeframes)
symbols = sorted(d.name for d in (base / tf).iterdir() if d.is_dir())
symbol = st.sidebar.selectbox("Symbol", symbols)
adir = base / tf / symbol

col1, col2 = st.columns(2)

# --- equity curve + gate status ---
perf_file = adir / "performance.json"
if perf_file.exists():
    perf = json.loads(perf_file.read_text())
    curve = perf.get("equity_curve", [])
    with col1:
        st.subheader("Equity curve (paper)")
        if curve:
            df = pd.DataFrame(curve, columns=["time", "equity"])
            df["time"] = pd.to_datetime(df["time"])
            st.line_chart(df.set_index("time")["equity"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe", f"{perf.get('sharpe') or 0:.2f}")
        c2.metric("Max drawdown", f"{(perf.get('max_drawdown') or 0) * 100:.1f}%")
        c3.metric("Trades", perf.get("trade_count", 0))

        from trading_bot.archive.store import Archive
        from trading_bot.live.gate import check_live_eligibility

        gate = check_live_eligibility(Archive(base, tf, symbol))
        if gate.allowed:
            st.success("LIVE GATE: PASSED")
        else:
            st.warning("LIVE GATE: BLOCKED\n\n" +
                       "\n".join(f"- {r}" for r in gate.reasons))

# --- ensemble weights ---
weights_file = adir / "weights.json"
if weights_file.exists():
    with col2:
        st.subheader("Ensemble weights (what the bot trusts)")
        wdata = json.loads(weights_file.read_text())
        wdf = pd.DataFrame(list(wdata.get("weights", {}).items()),
                           columns=["strategy", "weight"]).set_index("strategy")
        st.bar_chart(wdf)
        st.caption(f"Last updated: {wdata.get('last_updated', '?')}")

# --- trade log ---
trades_file = adir / "trades.csv"
st.subheader("Trade log")
if trades_file.exists():
    trades = pd.read_csv(trades_file)
    st.dataframe(trades.tail(50), use_container_width=True)
else:
    st.caption("No trades yet.")

# --- evolution population ---
pop_file = base / "evolution" / tf / "population.json"
if pop_file.exists():
    st.subheader("Trading floor (evolution pool)")
    pop = json.loads(pop_file.read_text())
    st.caption(f"Generation {pop.get('generation', 0)}")
    pdf = pd.DataFrame([
        {"trader": t["trader_id"], "kind": t["kind"],
         "fitness": t.get("fitness"),
         "survived": t.get("generations_survived", 0),
         **{f"p_{k}": v for k, v in t.get("params", {}).items()}}
        for t in pop.get("traders", [])
    ])
    st.dataframe(pdf.sort_values("fitness", ascending=False),
                 use_container_width=True)
