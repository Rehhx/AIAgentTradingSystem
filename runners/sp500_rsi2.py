"""
runners/sp500_rsi2.py
---------------------
Run the tuned RSI-2 mean-reversion strategy across the ENTIRE S&P 500, using
free split-adjusted daily bars from yfinance (the strategy is daily, so we are
not limited to the 20 local-parquet tickers).

Portfolio construction ("trade every S&P 500 name that signals"):
  - each day, go long every name whose RSI-2 signal is on (dip in an uptrend)
  - default: equal-weight across the names active that day (fully invested when
    >=1 signal, cash when none). Use --max-positions K to cap concurrent
    holdings at K (most-oversold first), each weighted 1/K.
  - 6 bps round-trip cost charged on daily turnover.

Metrics (Sharpe/$PnL/DD) come from this portfolio; win-rate / trade-count come
from the underlying per-name signal (every RSI-2 round-trip).

SURVIVORSHIP BIAS: uses today's index members over history -> optimistic. See
data/sp500.py.

Usage:
  python runners\\sp500_rsi2.py --limit 30            # quick test on 30 names
  python runners\\sp500_rsi2.py                        # full S&P 500
  python runners\\sp500_rsi2.py --max-positions 20     # cap at 20 holdings
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from data.sp500 import sp500_tickers, load_daily
from agents.daily_strategies import (
    sig_rsi2_meanrev, sleeve_returns, _rsi, DEPLOY_PARAMS, INITIAL_CAP, TRADING_DAYS,
)
from agents.risk_agent import RiskAgent
from config import RISK, RESULTS_DIR

PARAMS = DEPLOY_PARAMS["rsi2_meanrev"]      # rsi_period=2, entry=30, exit=50, sma=100


def build_panels(data: dict, params: dict):
    """wide frames aligned on the union of dates: held position (shifted),
    daily return, and decision-time RSI (for ranking). Also aggregate the
    per-name signal trades for win-rate / count."""
    pos, ret, rsi, all_trades = {}, {}, {}, []
    for t, d in data.items():
        raw = sig_rsi2_meanrev(d, params)
        pos[t] = raw.shift(1).fillna(0.0)               # enter next day
        ret[t] = d["close"].pct_change()
        rsi[t] = _rsi(d["close"], params.get("rsi_period", 2)).shift(1)
        _, trs = sleeve_returns(d, sig_rsi2_meanrev, params)
        all_trades.extend(trs)
    pos = pd.DataFrame(pos).sort_index()
    ret = pd.DataFrame(ret).reindex_like(pos)
    rsi = pd.DataFrame(rsi).reindex_like(pos)
    return pos.fillna(0.0), ret.fillna(0.0), rsi, all_trades


def portfolio(pos: pd.DataFrame, ret: pd.DataFrame, rsi: pd.DataFrame,
              max_positions: int, side_cost: float) -> pd.Series:
    """daily portfolio return net of turnover cost.

    Two sizing modes:
      max_positions == 0  -> DIVERSIFIED: each active name gets a fixed slot of
        1/N_universe, the rest is cash. Gross exposure = n_active / N, so the
        book is mostly cash and only leans in when many names are oversold at
        once. This is the risk-controlled default (matches the daily-book method).
      max_positions == K  -> CONCENTRATED: hold up to K most-oversold names at
        1/K each (cash if fewer than K signal). Up to 100% invested -> higher
        return and higher drawdown.
    """
    dates = pos.index
    N = pos.shape[1]
    W = pd.DataFrame(0.0, index=dates, columns=pos.columns)
    pos_np = pos.to_numpy()
    for i in range(len(dates)):
        active = np.where(pos_np[i] > 0)[0]
        if active.size == 0:
            continue
        if max_positions:
            if active.size > max_positions:
                rvals = rsi.iloc[i, active].to_numpy()
                active = active[np.argsort(np.nan_to_num(rvals, nan=999))[:max_positions]]
            W.iloc[i, active] = 1.0 / max_positions
        else:
            W.iloc[i, active] = 1.0 / N
    gross = (W * ret).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    return gross - turnover * side_cost


def metrics(net: pd.Series, trades: list) -> dict:
    net = net.fillna(0.0)
    eq = INITIAL_CAP * (1 + net).cumprod()
    final = float(eq.iloc[-1])
    sharpe = float(net.mean() / net.std() * np.sqrt(TRADING_DAYS)) if net.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    years = len(net) / TRADING_DAYS
    wins = sum(1 for t in trades if t["ret"] > 0)
    n = len(trades)
    return {
        "sharpe": round(sharpe, 3),
        "pnl_dollars": round(final - INITIAL_CAP, 2),
        "final_capital": round(final, 2),
        "total_return": round(final / INITIAL_CAP - 1, 4),
        "cagr": round((final / INITIAL_CAP) ** (1 / years) - 1, 4) if final > 0 else 0.0,
        "max_drawdown": round(dd, 4),
        "win_rate": round(wins / n, 4) if n else 0.0,
        "total_trades": n,
        "trades_per_year": round(n / years, 1),
        "exposure_pct": round(float((net != 0).mean()), 3),
        "_returns": net,
    }


def _sharpe(r):
    r = r.dropna()
    return round(float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)), 3) if r.std() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="test on first N tickers")
    ap.add_argument("--max-positions", type=int, default=0,
                    help="cap concurrent holdings (0 = equal-weight all active)")
    ap.add_argument("--low-vol", type=int, default=0,
                    help="keep only the K lowest-volatility S&P 500 names "
                         "(+ ETF anchors). 0 = full universe.")
    ap.add_argument("--anchors", default="SPY,QQQ,GLD",
                    help="index/ETF anchors always included with --low-vol")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--refresh", action="store_true", help="re-download from yfinance")
    args = ap.parse_args()

    tickers = sp500_tickers()
    if args.limit:
        tickers = tickers[:args.limit]
    print(f"\nRSI-2 across S&P 500 | {len(tickers)} tickers requested")
    print(f"params={PARAMS} | max_positions={args.max_positions or 'all-active'}"
          f"{' | low-vol top ' + str(args.low_vol) if args.low_vol else ''}")

    data = load_daily(tickers, start=args.start, refresh=args.refresh)
    if not data:
        print("No data loaded (yfinance unavailable?). Try again or --refresh.")
        return

    if args.low_vol:
        # rank by annualized daily-return volatility; keep the calmest K names
        vol = {t: d["close"].pct_change().std() * np.sqrt(TRADING_DAYS)
               for t, d in data.items()}
        calmest = sorted(vol, key=vol.get)[:args.low_vol]
        anchors = [a.strip().upper() for a in args.anchors.split(",") if a.strip()]
        anchor_data = load_daily(anchors, start=args.start, refresh=args.refresh)
        data = {t: data[t] for t in calmest}
        data.update(anchor_data)            # ETF anchors (may not be in the index)
        print(f"low-vol filter: kept {len(calmest)} calmest names + "
              f"{len(anchor_data)} anchors = {len(data)} total")
    print(f"Backtesting {len(data)} tickers with data ...\n")

    from agents.backtesting_agent import COMMISSION, SLIPPAGE
    side_cost = COMMISSION + SLIPPAGE
    pos, ret, rsi, trades = build_panels(data, PARAMS)
    net = portfolio(pos, ret, rsi, args.max_positions, side_cost)
    m = metrics(net, trades)

    # risk gate + walk-forward (70/30 split)
    verdict = RiskAgent().evaluate(m)
    s = int(len(net) * 0.7)
    train_sr, test_sr = _sharpe(net.iloc[:s]), _sharpe(net.iloc[s:])

    print(f"{'metric':16s}{'value':>16s}")
    print("-" * 32)
    for k in ["sharpe", "pnl_dollars", "cagr", "max_drawdown", "win_rate",
              "total_trades", "trades_per_year", "exposure_pct"]:
        v = m[k]
        if k == "pnl_dollars":
            print(f"{k:16s}{v:>15,.0f}$")
        elif k in ("cagr", "max_drawdown", "win_rate", "exposure_pct"):
            print(f"{k:16s}{v:>15.1%} ")
        else:
            print(f"{k:16s}{v:>16}")
    print("-" * 32)
    print(f"{'in-sample SR':16s}{train_sr:>16}")
    print(f"{'out-sample SR':16s}{test_sr:>16}")
    print(f"{'RISK GATE':16s}{('PASS' if verdict['passed'] else 'FAIL'):>16}")
    if verdict["failures"]:
        print(f"  fails: {verdict['failures']}")

    out = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "universe": "sp500",
        "n_tickers_with_data": len(data),
        "params": PARAMS,
        "max_positions": args.max_positions or "all-active",
        "cost_bps_round_trip": 6,
        "survivorship_bias_warning": "uses today's S&P 500 members over history",
        "metrics": {k: v for k, v in m.items() if not k.startswith("_")},
        "train_sharpe": train_sr, "test_sharpe": test_sr,
        "risk_passed": verdict["passed"], "risk_failures": verdict["failures"],
    }
    fp = Path(RESULTS_DIR) / "sp500_rsi2.json"
    fp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {fp}")


if __name__ == "__main__":
    main()
