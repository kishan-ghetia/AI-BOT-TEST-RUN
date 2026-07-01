# AI Trading Bot

Self-learning multi-strategy trading bot for **forex majors + top-10 crypto**,
across **daily / hourly / 5-minute** timeframes. Paper trading first; live
trading is locked behind a hard performance gate.

## How it decides

Five signal sources ("traders") each emit a signal in [-1, 1] with confidence:

| Strategy | What it does |
|---|---|
| `technical` | EMA crossover, RSI, MACD, Bollinger vote |
| `ml` | Gradient-boosting classifier on engineered features, retrained on a rolling window |
| `smc_ict` | Market structure (BOS/CHoCH), order blocks, fair value gaps, liquidity sweeps, premium/discount + OTE, session killzones |
| `fibonacci` | OTE (62–79%) retracement zones — also usable as a filter by other strategies |
| `news` | Finnhub economic calendar: surprise-vs-consensus signal + high-impact event risk filter (no-ops without `FINNHUB_API_KEY`) |
| `llm` | Stub for a future Claude-powered signal (not wired in) |

**Self-learning, three layers:**
1. **Adaptive ensemble weights** (multiplicative-weights/Hedge): strategies that call direction right gain influence; wrong ones lose it. Persisted per (timeframe, symbol).
2. **RL agent** (Q-learning with SGD function approximation): learns position sizing (0x/0.5x/1x/1.5x) and exit timing on top of the ensemble signal. Reward = risk-adjusted realized P&L.
3. **Evolution** (`evolve` command): a pool of N parameter-mutated strategy variants compete on backtests; the bottom third is fired each generation, winners breed mutated offspring.

**Risk:** 1% risk per trade sized off ATR stops, forex leverage/margin modeled,
20% max-drawdown circuit breaker, news blackout windows around high-impact events.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add keys when you have them; everything degrades gracefully

# backtest (also trains the learners); --synthetic if offline
python main.py backtest --symbol BTC-USD --timeframe daily
python main.py backtest --symbol EURUSD=X --timeframe daily --synthetic

# paper trade one cycle, or loop forever on each timeframe's own cadence
python main.py paper --symbols BTC-USD,EURUSD=X --timeframe hourly
python main.py paper --loop

# run the trading-floor evolution (fire bad traders, breed good ones)
python main.py evolve --symbol BTC-USD --timeframe daily --generations 3

# dashboard: equity curve, weights, trades, evolution pool
streamlit run dashboard.py

# live trading - REFUSES unless the gate passes (see below)
python main.py live --symbols BTC-USD --timeframe daily
```

## Live-trading hard gate

`live` will not construct a real broker unless, per (timeframe, symbol):
1. `LIVE_TRADING_ENABLED=true` explicitly set in `.env`
2. ≥ 4 weeks of **continuous** paper trading (gaps reset the clock)
3. Sharpe > 1.0 on the paper equity curve
4. Max drawdown < 20%

Enforced in `trading_bot/live/gate.py` — not just documentation.

## State layout (`data_store/`)

```
data_store/<timeframe>/<SYMBOL>/
  weights.json       # ensemble weights (what the bot trusts)
  model.pkl          # trained ML model
  rl_agent.json      # RL Q-function
  performance.json   # equity curve, Sharpe, max DD (feeds the live gate)
  trades.csv         # every closed trade
data_store/evolution/<timeframe>/population.json   # the trading floor
data_store/backtest_results/<run_id>/result.json   # backtests never touch paper state
```

## Notes

- Data via yfinance (no key). `--synthetic` / `SYNTHETIC_FALLBACK=true` gives
  deterministic simulated data for offline testing — never mistake those
  results for real performance.
- Broker adapters for ccxt (crypto) and OANDA (forex) are structurally
  complete and activate when keys are added to `.env`. `PaperBroker` is the
  default and needs nothing.
- Tests: `python -m pytest` — all synthetic, network calls are blocked in tests.
```
