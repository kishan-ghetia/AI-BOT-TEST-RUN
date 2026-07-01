"""Risk manager: ATR-based position sizing, forex leverage/margin math,
max-drawdown circuit breaker. Hard limits always win over RL/ensemble."""
from __future__ import annotations

from dataclasses import dataclass

import config
from trading_bot.data.symbols import is_forex


@dataclass
class SizingResult:
    size: float          # units of the asset
    notional: float      # size * price
    margin_required: float
    stop_price: float
    risk_amount: float   # capital at risk if stop hits


class RiskManager:
    def __init__(self, capital: float = config.INITIAL_CAPITAL,
                 risk_per_trade: float = config.RISK_PER_TRADE,
                 max_dd_pct: float = config.MAX_DRAWDOWN_PCT,
                 forex_leverage: float = config.FOREX_LEVERAGE,
                 crypto_leverage: float = config.CRYPTO_LEVERAGE):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_dd_pct = max_dd_pct
        self.forex_leverage = forex_leverage
        self.crypto_leverage = crypto_leverage

    def leverage_for(self, symbol: str) -> float:
        return self.forex_leverage if is_forex(symbol) else self.crypto_leverage

    def position_size(self, symbol: str, equity: float, price: float,
                      atr: float, side: str, stop_mult: float = 1.5) -> SizingResult:
        """Risk a fixed fraction of equity to an ATR-derived stop, capped by
        available margin at the symbol's leverage."""
        stop_distance = max(atr * stop_mult, price * 1e-4)
        risk_amount = equity * self.risk_per_trade
        size = risk_amount / stop_distance
        leverage = self.leverage_for(symbol)
        max_notional = equity * leverage
        notional = size * price
        if notional > max_notional:
            size = max_notional / price
            notional = max_notional
            risk_amount = size * stop_distance
        stop_price = price - stop_distance if side == "long" else price + stop_distance
        return SizingResult(
            size=size,
            notional=notional,
            margin_required=notional / leverage if leverage > 0 else notional,
            stop_price=stop_price,
            risk_amount=risk_amount,
        )

    def margin_required(self, notional: float, leverage: float) -> float:
        return notional / leverage if leverage > 0 else notional

    def check_drawdown_breaker(self, equity_curve: list[float]) -> bool:
        """True => trading halted: peak-to-trough drawdown hit the limit."""
        if len(equity_curve) < 2:
            return False
        peak = equity_curve[0]
        for eq in equity_curve:
            peak = max(peak, eq)
            if peak > 0 and (peak - eq) / peak >= self.max_dd_pct:
                return True
        return False

    def apply_rl_multiplier(self, sizing: SizingResult, multiplier: float,
                            equity: float, symbol: str,
                            price: float) -> SizingResult:
        """Scale base size by the RL action, re-clamping to margin limits."""
        leverage = self.leverage_for(symbol)
        size = sizing.size * multiplier
        notional = size * price
        max_notional = equity * leverage
        if notional > max_notional:
            size = max_notional / price
            notional = max_notional
        return SizingResult(
            size=size,
            notional=notional,
            margin_required=notional / leverage if leverage > 0 else notional,
            stop_price=sizing.stop_price,
            risk_amount=sizing.risk_amount * multiplier,
        )
