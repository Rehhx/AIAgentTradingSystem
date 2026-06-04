"""
runners/daily_vs_spy.py
-----------------------
Read-only daily scorecard: how did each account do TODAY versus the S&P 500?

Uses Alpaca's account.equity vs account.last_equity (equity at the prior close)
for the book's same-day return, and SPY's move since the prior close as the
benchmark. NO orders are placed — this only reads account state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings

warnings.filterwarnings("ignore")

from config import alpaca_keys, ALPACA_PAPER

NAMES = {1: "1 equity", 2: "2 futures", 3: "3 options"}


def account_today(account: int):
    key, secret = alpaca_keys(account)
    if not key or not secret:
        return None
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key=key, secret_key=secret, paper=ALPACA_PAPER)
        a = client.get_account()
        eq, last = float(a.equity), float(a.last_equity)
        return {"equity": eq, "last_equity": last,
                "ret": (eq / last - 1) if last else 0.0, "pl": eq - last}
    except Exception as e:
        return {"error": str(e)}


def spy_today():
    import yfinance as yf
    h = yf.Ticker("SPY").history(period="5d", auto_adjust=False)["Close"].dropna()
    return float(h.iloc[-1] / h.iloc[-2] - 1) if len(h) >= 2 else None


def main():
    spy = spy_today()
    print("=" * 60)
    print(f"DAILY SCORECARD vs S&P 500 (SPY)    paper={ALPACA_PAPER}")
    print("=" * 60)
    if spy is not None:
        print(f"  SPY today: {spy:+.2%}\n")
    print(f"  {'account':10s} {'today':>9s} {'vs SPY':>9s} {'P/L $':>12s} {'equity $':>12s}")
    print("  " + "-" * 54)
    for acc in (1, 2, 3):
        r = account_today(acc)
        if r is None:
            print(f"  {NAMES[acc]:10s}   (no keys configured)")
            continue
        if "error" in r:
            print(f"  {NAMES[acc]:10s}   error: {r['error'][:42]}")
            continue
        alpha = (r["ret"] - spy) if spy is not None else float("nan")
        print(f"  {NAMES[acc]:10s} {r['ret']:>+9.2%} {alpha:>+9.2%} "
              f"{r['pl']:>+12,.2f} {r['equity']:>12,.2f}")
    print()


if __name__ == "__main__":
    main()
