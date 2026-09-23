"""TrexStocks pipeline settings. Change numbers here, not in the other files."""
from pathlib import Path

VERSION = "5.0"
ALGORITHM_NAME = "Daily Picks v2 - Multi-Factor, Few-Weeks Horizon"
SCORE_MODEL = "Trex Score v2"     # shown on the site; old picks keep the model they used
PICKS_MODEL = "Daily Picks v2"

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
# Waiting for the backtest: also allow a stock up to 5% below its 200-day average
# if its 50-day trend is clearly improving. Leave False until tested.
TREND_RECOVERY_RULE = False
NO_REPEAT_DAYS = 5            # a stock cannot be picked again within 5 trading days

# Earnings (calendar days until the report). Affects Daily Picks only, never the Trex Score.
EARNINGS_WINDOW_DAYS = 14     # show an earnings badge inside this window
EARNINGS_EXCLUDE_DAYS = 3     # reporting within 3 days: not picked
EARNINGS_PENALTIES = [(7, 12), (14, 6)]  # 4-7 days: -12 points, 8-14 days: -6 points

# Entry timing. "v1" = best at RSI 50 (current). "v2" = best at RSI 55-65.
# Waiting for the backtest; leave "v1" until tested.
ENTRY_MODEL = "v1"

# ---------- Data coverage ----------
MIN_SCORE_COVERAGE = 0.75     # under 75% of the factor weight available: "Insufficient data"
LIMITED_DATA_WARNING = 0.90   # 75-90%: score shown with a "limited data" note
MIN_SECTOR_RANK_COUNT = 20    # fewer comparable stocks in a sector: rank against everyone

# ---------- Trex Score v2: one fixed formula for every stock ----------
TREX_WEIGHTS = {"trend": .20, "relStrength": .20, "entry": .10, "volume": .05,
                "risk": .10, "quality": .15, "value": .05, "growth": .15}
# Risk factor = volatility, beta and 1-year max drawdown on fixed scales (not ranks)
RISK_MIX = {"vol": .50, "beta": .25, "drawdown": .25}
# Descriptions by percentile of the Trex Score among every stock we cover
TIERS = [(85, "Very Strong"), (65, "Strong"), (35, "Middle"), (15, "Weak"), (0, "Very Weak")]

# ---------- Category rules ----------
# Internal key "speculative" is shown on the site as "Growth".
CATEGORY_LABELS = {"blueChip": "Blue Chip", "speculative": "Growth", "hiddenGem": "Hidden Gem"}
BLUE_CHIP_MIN_CAP_USD = 50e9
BLUE_CHIP_TSX_TOP_N = 25
GROWTH_MIN_CAP = {"USD": 2e9, "CAD": 2e9}
GROWTH_MIN_REVENUE_GROWTH = 0.15  # revenue growing 15%+ a year...
GROWTH_MIN_EARNINGS_GROWTH = 0.20  # ...or earnings growing 20%+
GEM_CAP_RANGE = {"USD": (1e9, 20e9), "CAD": (0.75e9, 10e9)}
GEM_MAX_ANALYSTS = 12             # analyst count must be known; unknown does not count

# ---------- Factor weights per category (each row adds to 1.0) ----------
# Starting hypotheses from the methodology review; the backtest will check them.
WEIGHTS = {
    "blueChip": {"trend": .20, "relStrength": .15, "entry": .10, "volume": .05,
                 "risk": .15, "quality": .20, "value": .10, "growth": .05},
    "speculative": {"trend": .20, "relStrength": .25, "entry": .10, "volume": .10,
                    "risk": .05, "quality": .05, "growth": .25},
    "hiddenGem": {"trend": .20, "relStrength": .20, "entry": .10, "volume": .10,
                  "risk": .05, "quality": .15, "value": .05, "growth": .15},
}

# Diversification
MAX_PER_SECTOR_DAILY = 3      # across all 9 picks
MAX_PER_SECTOR_TAB = 2        # within one index tab

# ---------- Portfolio ----------
REBALANCE_MONTHS = 6
PORTFOLIO_MIX = {"blueChip": 4, "speculative": 3, "hiddenGem": 1}
PORTFOLIO_MIN_TSX = 2
PORTFOLIO_MAX_PER_SECTOR = 2
PRICE_VERIFY_TOLERANCE = 0.005  # 0.5%

# Weekly copy of who is in each list, so future backtests use the real lists of the time
SNAPSHOT_DIR = STATE / "snapshots"

# ---------- Track record ----------
EVIDENCE_LABELS = [(50, "Early results"), (100, "Preliminary")]  # by distinct pick days

# ---------- Extras ----------
UNUSUAL_VOLUME_PACE = 3.0
TOP_MOVERS = 10

DISCLAIMER = ("FOR EDUCATIONAL PURPOSES ONLY. This is not financial advice. "
              "Past performance does not guarantee future results. Always do your own "
              "research and consult a financial advisor before making investment decisions.")
SCORE_DISCLOSURE = ("Scores rank measurable characteristics, not expected returns. "
                    "A higher score does not guarantee better future performance.")
