"""
Canonical data‑column names handled by the engine.

They are used for:
* prompt validation
* expression parsing
* schema checking at load time
"""

from enum import Enum


class DT(str, Enum):
    # ════════════════════════════════════════════════════════════════════
    # Mid‑price OHLC
    # ════════════════════════════════════════════════════════════════════
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"

    # Book‑edge quotes
    OPEN_ASK = "open_ask_price"
    CLOSE_ASK = "close_ask_price"
    OPEN_BID = "open_bid_price"
    CLOSE_BID = "close_bid_price"

    # Volume / trade counts
    VWAP = "vwap"
    BUY_VOL = "buy_volume"
    SELL_VOL = "sell_volume"
    TOTAL_VOL = "total_volume"
    BUY_N = "buy_trades_count"
    SELL_N = "sell_trades_count"
    TOTAL_N = "total_trades_count"

    # -------------------------------------------------------------------
    @classmethod
    def list(cls) -> list[str]:
        return [m.value for m in cls]
