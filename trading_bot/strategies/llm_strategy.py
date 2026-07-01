"""LLM strategy STUB - not wired into the ensemble yet.

Future: send recent price action + news headlines to the Anthropic API and
parse a directional view. Requires ANTHROPIC_API_KEY. Until implemented this
returns a neutral signal so it can never move the ensemble even if wired in.
"""
from __future__ import annotations

import pandas as pd

from trading_bot.strategies.base import Signal, Strategy


class LLMStrategy(Strategy):
    name = "llm"

    def generate_signal(self, df: pd.DataFrame, context: dict) -> Signal:
        # TODO: call Anthropic API with market context, parse response into
        # a directional score. Deliberately neutral until implemented.
        return self.neutral(df, context, reason="stub_not_implemented")
