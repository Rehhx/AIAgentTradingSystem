"""
runners/options_book.py
-----------------------
Account 3 = DEFINED-RISK LEAPS book. Implements the one genuinely good options
finding (see options_leverage.py): replace INDEX share exposure with deep-ITM
~1yr calls. Worst case is the premium (no margin call), the un-spent cash earns
T-bills, and historically this beat buy-and-hold on BOTH return and drawdown at
1.0x notional.

This builds the PREVIEW PLAN (which contract, how many, what it costs) for the
dashboard. Live submission is gated in the dashboard / execution_agent.submit_option.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from math import log, sqrt, exp, erf

# index sleeves we replace with LEAPS, and their share of the 1.0x exposure
LEAPS_UNDERLYINGS = {"SPY": 0.60, "QQQ": 0.40}
MONEYNESS = 0.90          # ~10% ITM -> ~0.8 delta
APPROX_DELTA = 0.80


def _bs_call(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    n = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return S * n(d1) - K * exp(-r * T) * n(d2)


def _vix():
    try:
        import yfinance as yf
        return float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]) / 100.0
    except Exception:
        return 0.18


def leaps_plan(agent, equity: float, leverage: float = 1.0) -> dict:
    """Return {ok, exposure, rows:[...], total_cost, pct_of_equity, cash_left, note}.
    Each row: underlying, symbol, strike, expiry, spot, est_premium, contracts,
    cost, source ('quote' or 'model')."""
    target_exposure = equity * leverage
    rows, total = [], 0.0
    iv = _vix()
    for u, share in LEAPS_UNDERLYINGS.items():
        notional = target_exposure * share
        if agent is not None and hasattr(agent, "find_leaps_contract"):
            c = agent.find_leaps_contract(u, moneyness=MONEYNESS)
        else:
            c = {"ok": False, "reason": "agent has no options method (stale app — fully restart streamlit)"}
        if not c.get("ok"):
            # model-only fallback so the preview still works without options access
            spot = _spot_fallback(u)
            if not spot:
                rows.append({"underlying": u, "ok": False, "reason": c.get("reason", "no data")})
                continue
            K = round(MONEYNESS * spot)
            est = _bs_call(spot, K, 1.0, iv) * 100.0
            contracts = max(1, round(notional / (spot * 100 * APPROX_DELTA)))
            cost = contracts * est
            total += cost
            rows.append({"underlying": u, "ok": True, "symbol": f"{u}~{K}C(~1yr) [model]",
                         "strike": K, "expiry": "~1yr", "spot": round(spot, 2),
                         "est_premium": round(est, 0), "contracts": contracts,
                         "cost": round(cost, 0), "source": "model"})
            continue
        spot = c["spot"]
        prem = (c.get("ask") or c.get("mid"))
        prem = prem * 100.0 if prem else _bs_call(spot, c["strike"], 1.0, iv) * 100.0
        contracts = max(1, round(notional / (spot * 100 * APPROX_DELTA)))
        cost = contracts * prem
        total += cost
        rows.append({"underlying": u, "ok": True, "symbol": c["symbol"],
                     "strike": c["strike"], "expiry": c["expiry"], "spot": round(spot, 2),
                     "est_premium": round(prem, 0), "contracts": contracts,
                     "cost": round(cost, 0),
                     "source": "quote" if (c.get("ask") or c.get("mid")) else "model"})
    return {
        "ok": any(r.get("ok") for r in rows),
        "exposure": target_exposure, "rows": rows, "total_cost": round(total, 0),
        "pct_of_equity": (total / equity if equity else 0.0),
        "cash_left": round(equity - total, 0),
        "note": ("Defined-risk: max loss = premium paid (no margin call). "
                 "Un-spent cash -> T-bills. 1.0x notional = same index exposure as holding shares."),
    }


def _spot_fallback(symbol):
    try:
        import yfinance as yf
        return float(yf.Ticker(symbol).history(period="5d")["Close"].iloc[-1])
    except Exception:
        return None


if __name__ == "__main__":
    # quick model-only preview without an Alpaca account
    print(leaps_plan(None, 100_000.0))
