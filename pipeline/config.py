"""TrexStocks pipeline settings. Change numbers here, not in the other files."""
from pathlib import Path

VERSION = "4.0"
ALGORITHM_NAME = "TrexPicks v4.0 - Multi-Factor, Few-Weeks Horizon"

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "state"

# Benchmarks shown on the index cards and used for relative strength
INDEX_SYMBOLS = {
    "sp500": {"yahoo": "^GSPC", "name": "S&P 500"},
    "nasdaq": {"yahoo": "^IXIC", "name": "NASDAQ Composite"},
    "tsx": {"yahoo": "^GSPTSE", "name": "S&P/TSX Composite"},
}
FX_SYMBOL = "CADUSD=X"  # price of 1 CAD in USD

# Market hours window (Eastern Time) in which refresh runs publish data.
# Runs a little past 4pm so the closing prices get captured.
REFRESH_START = (9, 25)
REFRESH_END = (16, 45)
PICKS_TIME = (10, 30)  # daily picks get locked on the first run after this

# ---------- Safety filters (a stock must pass all to be picked) ----------
MIN_PRICE = {"USD": 5.0, "CAD": 3.0}
MIN_DOLLAR_VOLUME = {"USD": 10_000_000, "CAD": 3_000_000}  # 50-day average
MIN_HISTORY_DAYS = 200
MAX_TODAY_MOVE = 8.0          # skip anything up or down more than 8% today
REQUIRE_ABOVE_200DMA = True   # only stocks in a long-term uptrend
NO_REPEAT_DAYS = 5            # a stock cannot be picked again within 5 trading days
EARNINGS_WINDOW_DAYS = 14     # calendar days (about 10 trading days)
EARNINGS_PENALTY = 15         # points taken off the final score

# ---------- Category rules ----------
BLUE_CHIP_MIN_CAP_USD = 50e9
BLUE_CHIP_TSX_TOP_N = 25
SPEC_MIN_CAP = {"USD": 2e9, "CAD": 2e9}
SPEC_MIN_REVENUE_GROWTH = 0.15
SPEC_VOL_PERCENTILE = 60      # or volatility in the top 40% of its index
GEM_CAP_RANGE = {"USD": (2e9, 20e9), "CAD": (1e9, 10e9)}
GEM_MAX_ANALYSTS = 10

# ---------- Factor weights per category (each row adds to 1.0) ----------
WEIGHTS = {
    "blueChip": {"trend": .20, "relStrength": .10, "entry": .15, "volume": .05,
                 "risk": .15, "quality": .20, "value": .15},
    "speculative": {"trend": .15, "relStrength": .25, "entry": .05, "volume": .15,
                    "quality": .05, "growth": .25, "breakout": .10},
    "hiddenGem": {"trend": .15, "relStrength": .15, "entry": .05, "volume": .15,
                  "quality": .15, "value": .05, "growth": .10, "breakout": .10,
                  "radar": .10},
}

# Diversification
MAX_PER_SECTOR_DAILY = 3      # across all 9 picks

# ---------- Portfolio ----------
REBALANCE_MONTHS = 6
PORTFOLIO_MIX = {"blueChip": 4, "speculative": 3, "hiddenGem": 1}
PORTFOLIO_MIN_TSX = 2
PORTFOLIO_MAX_PER_SECTOR = 2
PRICE_VERIFY_TOLERANCE = 0.005  # 0.5%

# ---------- Extras ----------
UNUSUAL_VOLUME_PACE = 3.0
TOP_MOVERS = 10

DISCLAIMER = ("FOR EDUCATIONAL PURPOSES ONLY. This is not financial advice. "
              "Past performance does not guarantee future results. Always do your own "
              "research and consult a financial advisor before making investment decisions.")
