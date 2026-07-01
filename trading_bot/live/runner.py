"""Live trading runner. The gate is checked BEFORE any real broker is
constructed; symbols that fail are skipped with printed reasons."""
from __future__ import annotations

import logging
from pathlib import Path

import config
from trading_bot.archive.store import Archive
from trading_bot.data.symbols import is_forex
from trading_bot.live.gate import check_live_eligibility
from trading_bot.scheduler import Scheduler

logger = logging.getLogger(__name__)


def run_live(timeframe: str, symbols: list[str],
             base_dir: Path = config.DATA_STORE_DIR) -> int:
    """Returns process exit code: 0 if anything traded live, 2 if fully blocked."""
    eligible: list[str] = []
    for symbol in symbols:
        archive = Archive(base_dir, timeframe, symbol)
        gate = check_live_eligibility(archive)
        if gate.allowed:
            eligible.append(symbol)
        else:
            print(f"[GATE BLOCKED] {timeframe}/{symbol}:")
            for reason in gate.reasons:
                print(f"  - {reason}")

    if not eligible:
        print("\nLive trading refused: no (timeframe, symbol) passed the gate.")
        return 2

    # Only now do we construct real brokers (fails clearly without keys).
    from trading_bot.execution.ccxt_broker import CcxtBroker
    from trading_bot.execution.oanda_broker import OandaBroker

    forex_syms = [s for s in eligible if is_forex(s)]
    crypto_syms = [s for s in eligible if not is_forex(s)]

    if forex_syms:
        broker = OandaBroker()
        Scheduler(base_dir, broker=broker).run_cycle(timeframe, forex_syms)
        print(f"Live cycle done (forex): {forex_syms}")
    if crypto_syms:
        broker = CcxtBroker()
        Scheduler(base_dir, broker=broker).run_cycle(timeframe, crypto_syms)
        print(f"Live cycle done (crypto): {crypto_syms}")
    return 0
