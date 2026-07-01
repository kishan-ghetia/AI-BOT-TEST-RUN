"""Composite SMC/ICT strategy: structure bias + confluence scoring from
order blocks, FVGs, liquidity sweeps, premium/discount + OTE, killzones."""
from __future__ import annotations

import pandas as pd

from trading_bot.strategies.base import Signal, Strategy
from trading_bot.strategies.smc_ict.fvg import find_fvgs, unfilled_fvgs
from trading_bot.strategies.smc_ict.liquidity import find_liquidity_pools, find_sweeps
from trading_bot.strategies.smc_ict.order_blocks import find_order_blocks, price_in_block
from trading_bot.strategies.smc_ict.sessions import session_weight
from trading_bot.strategies.smc_ict.structure import analyze_structure
from trading_bot.strategies.smc_ict.zones import premium_discount


class SMCICTStrategy(Strategy):
    name = "smc_ict"

    # SMC concepts describe recent structure; capping the analysis window
    # keeps per-bar cost constant instead of O(history^2) in backtests
    ANALYSIS_WINDOW = 200

    def __init__(self, lookback: int = 3, recent_bars: int = 10):
        self.lookback = lookback
        self.recent_bars = recent_bars  # how recent a sweep must be to count

    def warmup_bars(self) -> int:
        return 60

    def generate_signal(self, df: pd.DataFrame, context: dict) -> Signal:
        if len(df) < self.warmup_bars():
            return self.neutral(df, context, reason="warmup")
        df = df.tail(self.ANALYSIS_WINDOW)

        price = df["close"].iloc[-1]
        n = len(df)
        meta: dict = {}

        # 1. Directional bias from market structure
        state = analyze_structure(df, self.lookback)
        bias = {"bull": 1.0, "bear": -1.0}.get(state.trend, 0.0)
        meta["trend"] = state.trend
        score = bias * 0.4
        confluence = 0

        # 2. Premium/discount: longs from discount, shorts from premium
        zone, pos = premium_discount(df, self.lookback)
        meta["zone"] = zone
        if bias > 0 and zone == "discount":
            score += 0.2
            confluence += 1
        elif bias < 0 and zone == "premium":
            score -= 0.2
            confluence += 1
        elif (bias > 0 and zone == "premium") or (bias < 0 and zone == "discount"):
            score *= 0.5  # chasing into bad pricing: weaken

        # 3. Price inside a bias-aligned order block
        blocks = find_order_blocks(df)
        for block in blocks[-5:]:
            if price_in_block(price, block):
                if block.kind == "bullish" and bias >= 0:
                    score += 0.2
                    confluence += 1
                elif block.kind == "bearish" and bias <= 0:
                    score -= 0.2
                    confluence += 1
                meta["order_block"] = block.kind
                break

        # 4. Price inside an unfilled bias-aligned FVG
        gaps = unfilled_fvgs(df, find_fvgs(df))
        for gap in gaps[-5:]:
            if gap.bottom <= price <= gap.top:
                if gap.kind == "bullish" and bias >= 0:
                    score += 0.15
                    confluence += 1
                elif gap.kind == "bearish" and bias <= 0:
                    score -= 0.15
                    confluence += 1
                meta["fvg"] = gap.kind
                break

        # 5. Recent liquidity sweep = reversal fuel
        pools = find_liquidity_pools(df, lookback=self.lookback)
        sweeps = find_sweeps(df, pools)
        for sweep in sweeps:
            if n - sweep.index <= self.recent_bars:
                if sweep.kind == "low_sweep":
                    score += 0.25
                    confluence += 1
                else:
                    score -= 0.25
                    confluence += 1
                meta["sweep"] = sweep.kind
                break

        # 6. Session killzone weighting (confidence, not direction)
        weight, session = session_weight(df.index[-1], context.get("timeframe", "daily"))
        meta["session"] = session
        confidence = min(1.0, (0.3 + 0.15 * confluence)) * weight

        return Signal(
            symbol=context.get("symbol", "?"),
            timeframe=context.get("timeframe", "?"),
            timestamp=df.index[-1],
            strategy_name=self.name,
            value=score,
            confidence=confidence,
            metadata=meta,
        ).clamped()
