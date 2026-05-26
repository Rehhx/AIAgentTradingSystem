"""
config.py
---------
single source of truth for all system settings.
agents import from here — no hardcoded paths or values elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
DATA_DIR       = Path(os.getenv("DATA_DIR", r"C:\Users\pcagm\Downloads\StockData"))
RESULTS_DIR    = Path("results")
STRATEGIES_DIR = Path("strategies")
LOGS_DIR       = Path("logs")

# create dirs if they don't exist
for d in [RESULTS_DIR, STRATEGIES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# alpaca (paper trading)
# ---------------------------------------------------------------------------
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_PAPER      = True   # always True until we deliberately go live

# ---------------------------------------------------------------------------
# anthropic (claude agents)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# risk thresholds (risk_agent uses these)
# ---------------------------------------------------------------------------
RISK = {
    "min_sharpe":    0.8,
    "max_drawdown": -0.15,   # -15%
    "min_win_rate":  0.45,
    "min_trades":    50,
    "max_position_pct": 0.10,  # max 10% of portfolio in one position
}

# ---------------------------------------------------------------------------
# ml research settings
# ---------------------------------------------------------------------------
ML = {
    "default_window_days": 365,
    "train_test_split":    0.8,
    "models": ["xgboost", "lstm", "transformer"],
    "feature_windows": [5, 10, 20, 60],   # lookback periods for features
}

# ---------------------------------------------------------------------------
# data settings
# ---------------------------------------------------------------------------
DATA = {
    "timeframe":       "1min",
    "market_open":     "09:30",
    "market_close":    "16:00",
    "timezone":        "America/New_York",
    "resample_options": ["1min", "5min", "15min", "1h", "1d"],
}

# ---------------------------------------------------------------------------
# tickers to focus on by default
# (will auto-detect from parquet files, this is the fallback)
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "SPY", "QQQ", "TSLA", "NVDA", "AAPL",
    "GOOGL", "AMZN", "MSFT", "AMD", "JPM",
    "GLD", "GS", "NFLX", "CVNA", "TSM",
    "CAT", "SE", "CHWY", "NET", "TEM",
]
