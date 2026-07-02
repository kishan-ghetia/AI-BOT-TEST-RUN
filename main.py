"""AI trading bot CLI.

  python main.py backtest --symbol BTC-USD --timeframe daily [--start ... --end ...]
  python main.py paper    --symbols BTC-USD,EURUSD=X [--timeframe hourly] [--loop] [--max-cycles N]
  python main.py evolve   --symbol BTC-USD --timeframe daily [--generations N]
  python main.py live     --symbols ... --timeframe daily   (hard-gated)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from trading_bot.data.symbols import ALL_SYMBOLS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


def cmd_backtest(args) -> int:
    from trading_bot.backtest.engine import run_backtest
    from trading_bot.backtest.metrics import format_metrics
    from trading_bot.data.fetcher import get_ohlcv
    from trading_bot.data.finnhub_client import FinnhubClient
    from trading_bot.learning.adaptive_weights import AdaptiveWeights
    from trading_bot.learning.ensemble import Ensemble
    from trading_bot.learning.rl_agent import RLAgent
    from trading_bot.scheduler import default_strategies

    df = get_ohlcv(args.symbol, args.timeframe, start=args.start, end=args.end,
                   synthetic=args.synthetic)
    if df.empty:
        print(f"No data for {args.symbol} ({args.timeframe}) - "
              f"try --synthetic for offline testing")
        return 1
    label = "SYNTHETIC" if args.synthetic else "real"
    print(f"Backtesting {args.symbol} {args.timeframe} ({label} data): "
          f"{len(df)} bars ({df.index[0]} -> {df.index[-1]})")

    strategies = default_strategies(FinnhubClient(config.FINNHUB_API_KEY))
    ensemble = Ensemble(strategies, AdaptiveWeights([s.name for s in strategies]))
    result = run_backtest(df, ensemble, args.symbol, args.timeframe,
                          rl_agent=RLAgent(seed=42))

    print("\nResults:")
    print(format_metrics(result.metrics))
    print("\nFinal ensemble weights (what the bot learned to trust):")
    for name, w in sorted(result.final_weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {w:.3f}")

    # write to a separate backtest_results dir so backtests never pollute
    # the paper archive used by the live gate
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(config.DATA_STORE_DIR) / "backtest_results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps({
        "symbol": args.symbol, "timeframe": args.timeframe,
        "metrics": result.metrics, "final_weights": result.final_weights,
        "equity_curve": result.equity_curve,
        "trades": [{**t, "timestamp": str(t["timestamp"])} for t in result.trades],
    }, indent=2, default=str))
    print(f"\nSaved: {out_dir / 'result.json'}")
    return 0


def cmd_paper(args) -> int:
    from trading_bot.scheduler import Scheduler

    symbols = args.symbols.split(",") if args.symbols else ALL_SYMBOLS
    sched = Scheduler()
    if not args.loop:
        timeframes = [args.timeframe] if args.timeframe else ["daily"]
        for tf in timeframes:
            actions = sched.run_cycle(tf, symbols)
            _print_actions(tf, actions)
        return 0

    cycles = 0
    print("Paper loop started (ctrl-c to stop)...")
    while args.max_cycles is None or cycles < args.max_cycles:
        due = sched.due_timeframes()
        for tf in due:
            actions = sched.run_cycle(tf, symbols)
            _print_actions(tf, actions)
            cycles += 1
            if args.max_cycles is not None and cycles >= args.max_cycles:
                break
        time.sleep(args.tick_seconds)
    return 0


def _print_actions(tf: str, actions: list[dict]) -> None:
    for a in actions:
        line = f"[{tf}] {a.get('symbol')}: {a.get('action')}"
        if "signal" in a:
            line += f" (signal={a['signal']:+.3f}, conf={a['confidence']:.2f})"
        print(line)


def cmd_evolve(args) -> int:
    """Run the trading-floor evolution: N strategy variants compete on a
    backtest; losers are culled, winners breed."""
    from trading_bot.backtest.engine import run_backtest
    from trading_bot.data.fetcher import get_ohlcv
    from trading_bot.learning.adaptive_weights import AdaptiveWeights
    from trading_bot.learning.ensemble import Ensemble
    from trading_bot.learning.evolution import StrategyPool, validation_fitness

    symbols = args.symbols.split(",") if args.symbols else [args.symbol]
    frames = {}
    for sym in symbols:
        df = get_ohlcv(sym, args.timeframe, start=args.start, end=args.end,
                       synthetic=args.synthetic)
        if df.empty:
            print(f"No data for {sym} - try --synthetic for offline testing")
            return 1
        frames[sym] = df

    pool = StrategyPool(args.timeframe, Path(config.DATA_STORE_DIR),
                        pool_size=args.pool_size)
    for gen in range(args.generations):
        fitness: dict[str, float] = {}
        for trader in pool.traders:
            # portfolio fitness: a trader must generalize across markets,
            # not get lucky on one - mean of out-of-sample fitness per symbol
            scores = []
            for sym, df in frames.items():
                strat = trader.build()
                ens = Ensemble([strat], AdaptiveWeights([strat.name]))
                res = run_backtest(df, ens, sym, args.timeframe)
                scores.append(validation_fitness(res.equity_curve, args.timeframe))
            fitness[trader.trader_id] = sum(scores) / len(scores)
        pool.record_fitness(fitness)
        report = pool.evolve()
        print(f"Generation {report['generation']}: "
              f"best={report.get('best')} "
              f"fitness={report.get('best_fitness'):.3f} | "
              f"fired {len(report['culled'])}, hired {len(report['spawned'])}")
    print(f"\nPopulation saved to {pool.path}")
    return 0


def cmd_live(args) -> int:
    from trading_bot.live.runner import run_live

    symbols = args.symbols.split(",") if args.symbols else ALL_SYMBOLS
    return run_live(args.timeframe or "daily", symbols)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI trading bot")
    sub = p.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="backtest the ensemble on history")
    bt.add_argument("--symbol", required=True)
    bt.add_argument("--timeframe", default="daily", choices=config.TIMEFRAMES)
    bt.add_argument("--start")
    bt.add_argument("--end")
    bt.add_argument("--synthetic", action="store_true",
                    help="use simulated data (offline testing)")
    bt.set_defaults(func=cmd_backtest)

    pa = sub.add_parser("paper", help="paper-trade (simulated money)")
    pa.add_argument("--symbols", help="comma-separated; default = all")
    pa.add_argument("--timeframe", choices=config.TIMEFRAMES)
    pa.add_argument("--loop", action="store_true")
    pa.add_argument("--max-cycles", type=int, default=None)
    pa.add_argument("--tick-seconds", type=int, default=60)
    pa.set_defaults(func=cmd_paper)

    ev = sub.add_parser("evolve", help="evolve the strategy pool (fire/hire)")
    group = ev.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol")
    group.add_argument("--symbols", help="comma-separated basket for portfolio fitness")
    ev.add_argument("--timeframe", default="daily", choices=config.TIMEFRAMES)
    ev.add_argument("--start")
    ev.add_argument("--end")
    ev.add_argument("--generations", type=int, default=3)
    ev.add_argument("--pool-size", type=int, default=12)
    ev.add_argument("--synthetic", action="store_true",
                    help="use simulated data (offline testing)")
    ev.set_defaults(func=cmd_evolve)

    lv = sub.add_parser("live", help="live trading (hard-gated)")
    lv.add_argument("--symbols")
    lv.add_argument("--timeframe", choices=config.TIMEFRAMES)
    lv.set_defaults(func=cmd_live)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
