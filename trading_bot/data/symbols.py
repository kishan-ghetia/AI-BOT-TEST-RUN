"""Tradable symbol universe. Edit these lists to change what the bot trades."""

FOREX_MAJORS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "AUDUSD=X",
    "USDCAD=X",
    "NZDUSD=X",
    "EURGBP=X",
    "EURJPY=X",
    "GBPJPY=X",
]

CRYPTO_TOP10 = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "SOL-USD",
    "XRP-USD",
    "ADA-USD",
    "DOGE-USD",
    "TRX-USD",
    "AVAX-USD",
    "DOT-USD",
]

ALL_SYMBOLS = FOREX_MAJORS + CRYPTO_TOP10


def is_forex(symbol: str) -> bool:
    return symbol.endswith("=X")


def is_crypto(symbol: str) -> bool:
    return symbol.endswith("-USD")
