"""
runners/fundamental_screen.py
-----------------------------
Live fundamental QUALITY/VALUE screen (Finnhub) over a ticker set, and a demo of
its best use: a QUALITY FILTER on the momentum sleeve. High-momentum names that
are ALSO high-quality ("quality momentum") are a well-documented, more robust combo
than momentum alone; low-quality momentum names are the fragile ones to drop.

  python runners\fundamental_screen.py SPY,QQQ,MSFT,...
  python runners\fundamental_screen.py            # default: quality-10 + current momentum picks

HONEST: this is a LIVE screen (current fundamentals), usable as a filter/tilt on
today's holdings. It is NOT a backtested fundamental factor -- that needs paid
point-in-time data (see data/finnhub_fundamentals.py).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

from agents.daily_strategies import QUALITY_UNIVERSE, backtest_cross_sectional
from data.sp500 import sp500_tickers
from data.finnhub_fundamentals import fundamentals_frame, quality_score, FACTORS


def current_momentum_picks(k=10):
    """today's xs_dualmom top-k names (the live momentum holdings)."""
    try:
        res = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252,
                                       skip=21, k=k, market_filter=True)
        w = res.get("_weights")
        if w is not None and len(w):
            last = w.iloc[-1]
            return list(last[last > 0].index)
    except Exception:
        pass
    return []


def main():
    if len(sys.argv) > 1:
        tickers = [t.strip().upper() for t in sys.argv[1].split(",") if t.strip()]
        momentum = []
    else:
        momentum = current_momentum_picks(10)
        tickers = list(dict.fromkeys(QUALITY_UNIVERSE + momentum))

    print(f"fetching Finnhub fundamentals for {len(tickers)} names (cached) ...\n")
    df = fundamentals_frame(tickers)
    score = quality_score(df).sort_values(ascending=False)

    print("QUALITY/VALUE composite ranking (z-score of ROE, margin, liquidity, low debt, growth, low P/E):")
    print(f"  {'ticker':7s} {'score':>6s} {'ROE%':>7s} {'netMgn%':>8s} {'D/E':>7s} {'P/E':>7s} {'revGr%':>7s}")
    for t in score.index:
        r = df.loc[t]
        print(f"  {t:7s} {score[t]:+6.2f} {r['roeTTM']:7.1f} {r['netProfitMarginTTM']:8.1f} "
              f"{r['totalDebt/totalEquityQuarterly']:7.1f} {r['peTTM']:7.1f} {r['revenueGrowthTTMYoy']:7.1f}")

    if momentum:
        print(f"\nQUALITY FILTER on the {len(momentum)} live momentum picks "
              f"({', '.join(momentum)}):")
        mscore = score.reindex(momentum).dropna()
        keep = mscore[mscore > 0]
        drop = mscore[mscore <= 0]
        print(f"  PASS quality (score>0): {', '.join(keep.index) or '(none)'}")
        print(f"  FAIL quality (drop)   : {', '.join(drop.index) or '(none)'}")
        print("  -> 'quality momentum' = hold the PASS set; the FAIL names are momentum-only (more fragile).")

    print("\nNOTE: live screen on current fundamentals. A backtested fundamental factor")
    print("needs paid point-in-time data; use this to tilt/filter today's holdings.")


if __name__ == "__main__":
    main()
