"""
runners/daily_rebalance.py
--------------------------
Daily rebalancer that takes a daily book LIVE on Alpaca paper. This is the
Monday-deployment path for the strategy validated in runners/daily_book.py.

Each trading day after the close (or before the next open) you run this once:

  1. fetch recent daily bars for the universe (yfinance, split-adjusted; falls
     back to local parquet if offline)
  2. compute each sub-strategy's current long/flat position as of the last close
  3. turn positions into equal-weight target dollar weights for the book
  4. read current Alpaca paper positions + account equity
  5. diff target vs current and submit market orders for the deltas
     (DRY-RUN by default — prints the plan; pass --live to actually trade)

Books:
  rsi2_meanrev | donchian | trend_5020 | blended   (default: blended)

Usage:
  python runners\\daily_rebalance.py --book blended --universe SPY,QQQ,GLD,MSFT,JPM,GOOGL
  python runners\\daily_rebalance.py --book blended --live          # actually trades paper
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    STRATEGIES_DAILY, CANDIDATE_STRATEGIES, DEPLOY_PARAMS,
    DEFAULT_UNIVERSE, QUALITY_UNIVERSE, daily_bars,
)
from agents.execution_agent import ExecutionAgent

# all sig functions available to the rebalancer (core + promoted candidates)
ALL_SIGNALS = {**STRATEGIES_DAILY, **CANDIDATE_STRATEGIES}

# each book maps sub-strategy -> capital weight (weights sum to 1).
BOOKS = {
    "rsi2_meanrev": {"rsi2_meanrev": 1.0},
    "donchian":     {"donchian": 1.0},
    "trend_5020":   {"trend_5020": 1.0},
    "blended":      {"rsi2_meanrev": 1/3, "donchian": 1/3, "trend_5020": 1/3},
    # trend-tilted blend: ~12% CAGR, Sharpe ~1.25, -17% DD (needs DD limit ~-18%)
    "trend_tilt":   {"trend_5020": 0.5, "rsi2_meanrev": 0.5},
    # defensive: adds the uncorrelated turn-of-month sleeve -> lowest DD (~-8%)
    "defensive":    {"rsi2_meanrev": 0.25, "donchian": 0.25,
                     "trend_5020": 0.25, "turn_of_month": 0.25},
    # blended_plus: core-3 + cross-sectional dual-momentum -> Sharpe 1.26,
    # 11.5% CAGR, -13.9% DD (passes risk; best return-per-risk found so far)
    "blended_plus": {"rsi2_meanrev": 0.25, "donchian": 0.25,
                     "trend_5020": 0.25, "xs_dualmom": 0.25},
    # portfolio: the auto-allocator's RISK-PARITY weights over the 4 admitted
    # strategies (inverse-vol). With --vol-target 0.16 --max-leverage 1.6 -> ~16.2%
    # CAGR, -13.0% DD, Sharpe 1.38, 5/5 walk-forward folds. The 15-20% target book.
    "portfolio":    {"rsi2_meanrev": 0.41, "donchian": 0.32,
                     "trend_5020": 0.18, "xs_dualmom": 0.09},
    # regime_adaptive: weights + leverage shift by SPY regime (see _detect_regime).
    # AGGRESSIVE growth book — leverage is OPT-IN via --max-leverage (default 1.0
    # = regime tilting, no leverage). With --max-leverage 1.5: ~20% CAGR, ~-18% DD.
    "regime_adaptive": {"_regime": 1.0},
    # pead: post-earnings-drift — scans the full S&P 500 for earnings-gap beats,
    # holds each for its 60-day drift window (event-driven; auto buy/hold/sell).
    "pead": {"pead": 1.0},
    # portfolio_div: the portfolio book (85%) + a 15% event-driven PEAD satellite.
    # Best risk-adjusted config: Sharpe 1.47, 16.0% CAGR, -12.3% DD, 5/5 folds.
    "portfolio_div": {"rsi2_meanrev": 0.35, "donchian": 0.27, "trend_5020": 0.15,
                      "xs_dualmom": 0.08, "pead": 0.15},
    # portfolio_rec: core sleeves + a 20% RECOVERY sleeve that captures bull-run
    # snapbacks (early-2019, spring-2020). Lifts 2018-2020 to +8.4% while raising
    # Sharpe to 1.43 and CAGR to 17.1% at -14.1% DD. Run with --vol-target 0.15.
    "portfolio_rec": {"rsi2_meanrev": 0.32, "donchian": 0.24, "trend_5020": 0.16,
                      "xs_dualmom": 0.08, "recovery": 0.20},
    # portfolio_full: 7 sleeves — core + recovery (bull capture) + PEAD (event
    # smoothing) + defensive lowvol (bear/vol ballast: holds the 30 lowest-vol S&P
    # names while SPY > 200d, rotates to BIL otherwise). The six price sleeves are
    # scaled to 90% and lowvol takes 10%. Sharpe 1.53, 18.4% CAGR, -13.4% DD, 5/5
    # folds. Run with --vol-target 0.17 --max-leverage 1.8.
    "portfolio_full": {"rsi2_meanrev": 0.252, "donchian": 0.198, "trend_5020": 0.126,
                       "xs_dualmom": 0.072, "recovery": 0.162, "pead": 0.090,
                       "lowvol_def": 0.10},
    # managed_futures: standalone long/SHORT trend (CTA) book for ACCOUNT 2 -- crisis
    # alpha. Diversified time-series momentum across 10 asset ETFs, vol-targeted.
    # Profits in macro bears (2022 +6.5% vs S&P -18%); choppy in calm bulls. REQUIRES
    # shorting + margin on the Alpaca account; uses whole-share orders (no fractional
    # shorts). Run on account 2: --book managed_futures --account 2 --whole-shares
    "managed_futures": {"_mf": 1.0},
}

MF_MARKETS = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "VNQ"]

PEAD_PARAMS = {"gap_pct": 0.05, "vol_mult": 2.0, "hold_days": 60}

# per-regime (per-ticker sleeve weights, xs_dualmom alloc, leverage)
REGIME_WEIGHTS = {
    "bull_calm": ({"trend_5020": 0.30, "rsi2_meanrev": 0.15, "donchian": 0.15}, 0.40, 1.5),
    "bull_vol":  ({"rsi2_meanrev": 0.45, "donchian": 0.20, "trend_5020": 0.20}, 0.15, 1.0),
    "bear":      ({}, 0.0, 1.0),   # defensive: gold + cash, handled below
}


def _detect_regime(source: str) -> str:
    """SPY 200-day trend + 20-day realized vol -> bull_calm / bull_vol / bear."""
    spy = fetch_daily("SPY", source=source)["close"]
    if len(spy) < 210:
        return "bull_calm"
    trend_up = float(spy.iloc[-1]) > float(spy.rolling(200).mean().iloc[-1])
    vol = float(spy.pct_change().rolling(20).std().iloc[-1]) * (252 ** 0.5)
    if not trend_up:
        return "bear"
    return "bull_vol" if vol >= 0.16 else "bull_calm"


def regime_status(source: str = "auto") -> None:
    """Print the current market regime + the indicators behind it, so the
    bull/bear transition is visible each run."""
    try:
        spy = fetch_daily(XS_MKT, source=source)["close"]
        last = float(spy.iloc[-1])
        sma50 = float(spy.rolling(50).mean().iloc[-1])
        sma200 = float(spy.rolling(200).mean().iloc[-1])
        vol = float(spy.pct_change().rolling(20).std().iloc[-1]) * (252 ** 0.5)
        reg = _detect_regime(source)
        print(f"  [regime] SPY {last:.0f} | 50d {sma50:.0f} | 200d {sma200:.0f} "
              f"({last/sma200 - 1:+.1%}) | 20d vol {vol:.0%}  ->  {reg.upper()}")
        notes = {
            "bear":      "DEFENSIVE: trend/momentum -> cash, recovery dormant, cash parked in T-bills",
            "bull_vol":  "RISK-ON (elevated vol): vol-target de-risks, mean-reversion favored",
            "bull_calm": "RISK-ON: full exposure, leverage available, recovery sleeve active",
        }
        print(f"    -> {notes.get(reg, '')}")
    except Exception:
        pass


def fetch_daily(ticker: str, lookback_days: int = 400, source: str = "auto") -> pd.DataFrame:
    """recent split-adjusted daily bars. yfinance first (fresh, adjusted),
    parquet fallback (stale but offline-safe)."""
    if source in ("auto", "yfinance"):
        try:
            import yfinance as yf
            raw = yf.Ticker(ticker).history(
                period=f"{lookback_days + 50}d", interval="1d", auto_adjust=True)
            if not raw.empty:
                raw = raw.rename(columns={"Open": "open", "High": "high",
                                          "Low": "low", "Close": "close",
                                          "Volume": "volume"})
                idx = raw.index
                raw.index = idx.tz_convert("UTC") if idx.tz else idx.tz_localize("UTC")
                return raw[["open", "high", "low", "close", "volume"]].dropna()
        except Exception as e:
            if source == "yfinance":
                raise
            print(f"  [warn] yfinance failed for {ticker} ({e}); using local parquet")
    return daily_bars(ticker).tail(lookback_days + 50)


# cross-sectional dual-momentum sleeve config (matches backtest_cross_sectional)
XS_LOOKBACK, XS_SKIP, XS_MKT, XS_SMA = 252, 21, "SPY", 200
LOWVOL_K, LOWVOL_VOLWIN = 30, 60          # defensive low-vol sleeve: 30 lowest 60d-vol names


def _managed_futures_weights(source, target_vol, max_lev):
    """today's SIGNED weights for the managed-futures (L/S trend) book: diversified
    time-series momentum (avg sign of 1/3/12-mo returns), inverse-vol risk weighted,
    vol-targeted. Returns (weights, last_price, detail) with negative weights = shorts."""
    closes, last_price = {}, {}
    for t in MF_MARKETS:
        try:
            c = fetch_daily(t, source=source)["close"]
            if len(c) > 300:
                closes[t] = c
                last_price[t] = float(c.iloc[-1])
        except Exception as e:
            print(f"  [mf] skip {t}: {e}")
    C = pd.DataFrame(closes).sort_index()
    R = C.pct_change()
    sig = (np.sign(C / C.shift(21) - 1) + np.sign(C / C.shift(63) - 1) + np.sign(C / C.shift(252) - 1)) / 3.0
    vol = R.rolling(60).std()
    raw = sig / vol
    Wg = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0)      # gross ~1 each day
    port = (Wg.shift(1) * R).sum(axis=1)
    rv = float(port.tail(20).std() * np.sqrt(252))
    scale = min(max_lev, target_vol / rv) if rv > 0 else 1.0
    today = (Wg.iloc[-1] * scale).dropna()
    weights = {t: float(today[t]) for t in today.index if abs(today[t]) > 1e-4}
    detail = {t: [("mf_short" if w < 0 else "mf_long")] for t, w in weights.items()}
    print(f"  [managed-futures] {len(weights)} positions | vol-target {target_vol:.0%} "
          f"-> leverage {scale:.2f}x | gross {sum(abs(w) for w in weights.values()):.0%}")
    return weights, last_price, detail


def target_weights(book: str, universe: list, source: str = "auto",
                   xs_universe: str = "same", vol_target_pct: float = 0.0,
                   max_lev: float = 1.0, park_cash: str = "BIL",
                   early_warning: bool = True, name_cap: float = 0.0,
                   crypto_weight: float = 0.0) -> tuple[dict, dict, dict]:
    """returns (weights, last_price, detail). Per-ticker sleeves give each long
    name alloc/N. The 'xs_dualmom' sleeve ranks a (possibly larger) universe by
    12-1 momentum and gives the top-K names alloc/K (cash when SPY < its 200d
    SMA). vol_target_pct>0 scales the whole book to that annualized vol."""
    if book == "managed_futures":
        return _managed_futures_weights(source, vol_target_pct or 0.12,
                                        max_lev if max_lev > 1.0 else 1.5)

    leverage, bear = 1.0, False
    if book == "regime_adaptive":
        regime = _detect_regime(source)
        w, xs_a, reg_lev = REGIME_WEIGHTS[regime]
        strat_weights = dict(w)
        xs_alloc = xs_a
        leverage = min(reg_lev, max_lev) if max_lev > 1.0 else 1.0  # leverage OPT-IN
        bear = (regime == "bear")
        xs_universe = "sp500"                # rank the cross-sectional sleeve index-wide
        print(f"  [regime] {regime} | sleeve leverage {leverage:.2f}x (cap {max_lev}) | xs=S&P500")
    else:
        strat_weights = dict(BOOKS[book])    # {strategy: capital weight}
        xs_alloc = strat_weights.pop("xs_dualmom", 0.0)
    pead_alloc = strat_weights.pop("pead", 0.0)
    lowvol_alloc = strat_weights.pop("lowvol_def", 0.0)
    n = len(universe)
    weights = {t: 0.0 for t in universe}
    last_price, detail, closes = {}, {t: [] for t in universe}, {}
    for t in universe:
        try:
            d = fetch_daily(t, source=source)
        except Exception as e:
            print(f"  [skip] {t}: {e}")
            continue
        if len(d) < 220:
            print(f"  [skip] {t}: only {len(d)} daily bars (<220 warmup)")
            continue
        last_price[t] = float(d["close"].iloc[-1])
        closes[t] = d["close"]
        for s, alloc in strat_weights.items():
            pos = ALL_SIGNALS[s](d, DEPLOY_PARAMS.get(s) or None)
            if float(pos.iloc[-1]) > 0:      # position as of the last close
                weights[t] += alloc * (1.0 / n)
                detail[t].append(s)

    # cross-sectional dual-momentum sleeve (rank over `universe` or full S&P 500)
    if xs_alloc > 0:
        market_ok = True
        try:
            mkt = fetch_daily(XS_MKT, source=source)["close"]
            market_ok = float(mkt.iloc[-1]) > float(mkt.rolling(XS_SMA).mean().iloc[-1])
        except Exception:
            pass
        if market_ok:
            if xs_universe == "sp500":
                from data.sp500 import sp500_tickers
                rank_closes, k = {}, 10
                for t in sp500_tickers():
                    try:
                        rank_closes[t] = daily_bars(t)["close"]   # adjusted, cached
                    except Exception:
                        continue
            else:
                rank_closes, k = closes, 3
            scores = {t: float(c.iloc[-1 - XS_SKIP] / c.iloc[-1 - XS_SKIP - XS_LOOKBACK] - 1)
                      for t, c in rank_closes.items() if len(c) > XS_SKIP + XS_LOOKBACK + 1}
            for t in sorted(scores, key=scores.get, reverse=True)[:k]:
                weights[t] = weights.get(t, 0.0) + xs_alloc * (1.0 / k)
                detail.setdefault(t, []).append("xs_dualmom")
                if t not in last_price:
                    last_price[t] = float(rank_closes[t].iloc[-1])
                    closes[t] = rank_closes[t]

    # PEAD sleeve: scan the full S&P 500, hold every name currently inside its
    # post-earnings drift window, equal-weight. Reconciler buys fresh beats and
    # sells names whose window has expired.
    if pead_alloc > 0:
        from data.sp500 import sp500_tickers
        PEAD_MAX = 25                              # cap so positions clear the no-trade band
        active = []
        for t in sp500_tickers():
            try:
                d = daily_bars(t)
            except Exception:
                continue
            if len(d) <= 80:
                continue
            pos = ALL_SIGNALS["pead"](d, PEAD_PARAMS).to_numpy()
            if pos[-1] > 0:
                run = 0                            # days since the beat (shorter = fresher drift)
                for v in pos[::-1]:
                    if v > 0:
                        run += 1
                    else:
                        break
                active.append((run, t, float(d["close"].iloc[-1])))
        active.sort()                              # freshest beats first
        active = active[:PEAD_MAX]
        if active:
            per = pead_alloc / len(active)
            for run, t, px in active:
                weights[t] = weights.get(t, 0.0) + per
                detail.setdefault(t, []).append("pead")
                if t not in last_price:
                    last_price[t] = px
            print(f"  [pead] holding {len(active)} freshest post-earnings-drift names")

    # lowvol sleeve (defensive): hold the K lowest realized-vol S&P 500 names,
    # equal weight, ONLY while SPY > its 200-day. In a bear (SPY < 200d) this alloc
    # stays idle and is swept to BIL by park_cash -> the sleeve sits out slow bears
    # (2018/2022 finished flat-to-up) instead of riding them down.
    if lowvol_alloc > 0:
        lv_on = True
        try:
            mkt = fetch_daily(XS_MKT, source=source)["close"]
            lv_on = float(mkt.iloc[-1]) > float(mkt.rolling(XS_SMA).mean().iloc[-1])
        except Exception:
            pass
        if lv_on:
            from data.sp500 import sp500_tickers
            vols = {}
            for t in sp500_tickers():
                try:
                    c = daily_bars(t)["close"]
                except Exception:
                    continue
                if len(c) < LOWVOL_VOLWIN + 5:
                    continue
                rv = float(c.pct_change().rolling(LOWVOL_VOLWIN).std().iloc[-1])
                if rv == rv and rv > 0:
                    vols[t] = (rv, c)
            picks = sorted(vols, key=lambda t: vols[t][0])[:LOWVOL_K]
            for t in picks:
                weights[t] = weights.get(t, 0.0) + lowvol_alloc * (1.0 / len(picks))
                detail.setdefault(t, []).append("lowvol_def")
                if t not in last_price:
                    c = vols[t][1]
                    last_price[t] = float(c.iloc[-1]); closes[t] = c
            print(f"  [lowvol] holding {len(picks)} lowest-vol S&P names (SPY > 200d)")
        else:
            print("  [lowvol] SPY < 200d -> lowvol sleeve in cash (BIL)")

    # regime_adaptive: bear -> defensive gold+cash; else apply conditional leverage
    if book == "regime_adaptive":
        if bear:
            weights = {t: 0.0 for t in weights}
            if "GLD" in last_price:
                weights["GLD"] = 0.40
                detail.setdefault("GLD", []).append("regime:bear-gold")
            else:
                print("  [regime] bear -> 100% cash (GLD not in universe)")
        elif leverage != 1.0:
            weights = {t: w * leverage for t, w in weights.items()}

    # crypto sleeve (OPT-IN, default off): absolute-momentum on BTC/ETH. Carves out
    # crypto_weight from the book (scales the rest down) and allocates to each crypto
    # whose 6-month trend is up, else leaves it idle -> BIL. GOVERNANCE-GATED: only
    # enable with explicit board sign-off; sized small (~5%) to stay inside the gate.
    if crypto_weight > 0:
        for t in list(weights):
            weights[t] *= (1 - crypto_weight)
        on = []
        for sym in ("BTC-USD", "ETH-USD"):
            try:
                d = fetch_daily(sym, source=source)
                if float(ALL_SIGNALS["abs_momentum"](d, {"lookback": 126}).iloc[-1]) > 0:
                    on.append((sym, d["close"]))
            except Exception as e:
                print(f"  [crypto] {sym} skipped: {e}")
        per = crypto_weight / 2.0          # equal-weight over the 2-name crypto universe
        for sym, c in on:
            weights[sym] = weights.get(sym, 0.0) + per
            detail.setdefault(sym, []).append("crypto_mom")
            last_price[sym] = float(c.iloc[-1]); closes[sym] = c
        if on:
            print(f"  [crypto] momentum ON ({crypto_weight:.0%} sleeve): {', '.join(s for s, _ in on)}")
        else:
            print(f"  [crypto] momentum OFF -> {crypto_weight:.0%} stays in cash/BIL")

    # volatility-targeting overlay: scale the whole book to target annualized vol
    if vol_target_pct > 0:
        held = {t: w for t, w in weights.items() if w > 0 and t in closes}
        if held:
            sub = pd.concat([closes[t].pct_change() for t in held], axis=1, sort=True)
            wv = np.array([held[t] for t in held], float); wv /= wv.sum()
            port_r = (sub.fillna(0.0) * wv).sum(axis=1).tail(60)
            rv = float(port_r.std() * np.sqrt(252))
            scale = min(max_lev, vol_target_pct / rv) if rv > 0 else 1.0
            weights = {t: w * scale for t, w in weights.items()}
            print(f"  [vol-target {vol_target_pct:.0%}] recent vol {rv:.0%} "
                  f"-> exposure scale {scale:.2f}x")

    # early-warning de-risk: cut exposure to 60% when SPY breaks its 50-day AND
    # 20-day vol > 20% -- front-runs the lagging 200-day bear signal (improves
    # Sharpe 1.45->1.48 and DD -13.8%->-11.7% in backtest).
    if early_warning:
        try:
            spy = fetch_daily(XS_MKT, source=source)["close"]
            below50 = float(spy.iloc[-1]) < float(spy.rolling(50).mean().iloc[-1])
            volspike = float(spy.pct_change().rolling(20).std().iloc[-1]) * (252 ** 0.5) > 0.20
            if below50 and volspike:
                weights = {t: w * 0.6 for t, w in weights.items()}
                print("  [early-warning] SPY < 50d AND vol > 20% -> de-risked to 60%")
        except Exception:
            pass

    # single-name concentration cap (risk governance): no individual ticker above
    # name_cap of the book; freed exposure is swept to T-bills below. A forward
    # safety rail -- currently near-non-binding (largest name ~10%) but caps any one
    # stock from dominating if its sleeve signals stack or leverage rises.
    if name_cap and name_cap > 0:
        capped = [t for t, w in weights.items()
                  if w > name_cap and t != (park_cash or "")]
        for t in capped:
            weights[t] = name_cap
        if capped:
            print(f"  [name-cap] capped {len(capped)} name(s) at {name_cap:.0%}: {', '.join(sorted(capped))}")

    # park idle cash in a T-bill ETF so uninvested capital earns ~yield, not 0%.
    # Self-reinforcing for lean years: defensive periods hold more cash -> more yield.
    if park_cash:
        invested = sum(w for w in weights.values() if w > 0)
        idle = round(1.0 - invested, 4)
        if idle > 0.01:
            try:
                weights[park_cash] = weights.get(park_cash, 0.0) + idle
                detail.setdefault(park_cash, []).append("cash_yield")
                if park_cash not in last_price:
                    last_price[park_cash] = float(fetch_daily(park_cash, source=source)["close"].iloc[-1])
                print(f"  [cash-yield] parking {idle:.0%} idle cash in {park_cash} (T-bill yield)")
            except Exception as e:
                print(f"  [warn] could not park cash in {park_cash}: {e}")
    return weights, last_price, detail


def get_equity(agent: ExecutionAgent, override: float | None) -> tuple[float, str]:
    if override:
        return override, "override"
    if not agent.simulated and agent.client is not None:
        try:
            acct = agent.client.get_account()
            return float(acct.equity), "alpaca"
        except Exception as e:
            print(f"  [warn] could not read Alpaca equity ({e}); using $100k")
    return 100_000.0, "default"


def current_positions(agent: ExecutionAgent) -> dict:
    return {p["symbol"]: float(p["qty"]) for p in agent.get_positions()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="blended", choices=list(BOOKS))
    ap.add_argument("--universe", default="quality",
                    help="comma list, or 'quality' (10 locked names), or 'default' (8)")
    ap.add_argument("--live", action="store_true", help="actually submit paper orders")
    ap.add_argument("--notional", type=float, default=None,
                    help="override account equity used for sizing (dry-run default $100k)")
    ap.add_argument("--source", default="auto", choices=["auto", "yfinance", "parquet"])
    ap.add_argument("--xs-universe", default="same", choices=["same", "sp500"],
                    help="ranking universe for the cross-sectional sleeve")
    ap.add_argument("--vol-target", type=float, default=0.0,
                    help="annualized vol target (e.g. 0.12); 0 = off")
    ap.add_argument("--max-leverage", type=float, default=1.0,
                    help="max exposure when vol-targeting (1.0 = de-risk only)")
    ap.add_argument("--whole-shares", action="store_true",
                    help="use integer shares (default: fractional/notional dollar orders)")
    ap.add_argument("--min-order", type=float, default=250.0,
                    help="no-trade band: skip reconciling orders smaller than $X")
    ap.add_argument("--park-cash", default="BIL",
                    help="park idle cash in this T-bill ETF for yield ('' to disable)")
    ap.add_argument("--no-early-warning", action="store_true",
                    help="disable the SPY-50d/vol early-warning de-risk overlay")
    ap.add_argument("--name-cap", type=float, default=0.10,
                    help="max weight per single name (risk governance); 0 = off")
    ap.add_argument("--crypto-sleeve", action="store_true",
                    help="OPT-IN: enable the BTC/ETH momentum sleeve (needs board sign-off)")
    ap.add_argument("--crypto-weight", type=float, default=0.05,
                    help="crypto sleeve weight when --crypto-sleeve is set (default 5%%)")
    ap.add_argument("--account", type=int, default=1, choices=[1, 2],
                    help="which Alpaca paper account (1=default keys, 2=ALPACA_*_2 keys)")
    args = ap.parse_args()

    if args.universe.strip().lower() == "quality":
        universe = list(QUALITY_UNIVERSE)
    elif args.universe.strip().lower() == "default":
        universe = list(DEFAULT_UNIVERSE)
    else:
        universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    mode = "LIVE (paper)" if args.live else "DRY-RUN"
    print(f"\nDaily rebalance | book={args.book} | {mode}")
    print(f"Universe ({len(universe)}): {', '.join(universe)}\n")

    if args.account == 2:
        from config import ALPACA_API_KEY_2, ALPACA_API_SECRET_2
        agent = ExecutionAgent(api_key=ALPACA_API_KEY_2, api_secret=ALPACA_API_SECRET_2)
    else:
        agent = ExecutionAgent()
    print(f"Alpaca account: #{args.account}")
    if args.live and agent.simulated:
        print("  [warn] Alpaca creds missing -- --live will only SIMULATE fills.")

    regime_status(args.source)             # show bull/bear regime + the indicators

    weights, last_price, detail = target_weights(
        args.book, universe, args.source, xs_universe=args.xs_universe,
        vol_target_pct=args.vol_target, max_lev=args.max_leverage,
        park_cash=(args.park_cash.strip().upper() or None),
        early_warning=(not args.no_early_warning), name_cap=args.name_cap,
        crypto_weight=(args.crypto_weight if args.crypto_sleeve else 0.0))
    equity, esrc = get_equity(agent, args.notional)
    held = current_positions(agent)

    print(f"Account equity: ${equity:,.0f} (source: {esrc})")
    invested = sum(weights.values())
    gross = sum(abs(w) for w in weights.values())
    if gross > invested + 1e-9:                 # book has shorts (e.g. managed_futures)
        print(f"Net exposure: {invested:+.0%} | gross (long+short): {gross:.0%}\n")
    else:
        print(f"Target invested: {invested:.0%} | cash: {max(0, 1 - invested):.0%}\n")

    sizing = "whole-share" if args.whole_shares else "fractional (notional $)"
    print(f"Sizing: {sizing} | no-trade band: ${args.min_order:,.0f}\n")
    hdr = (f"{'ticker':7s} {'signals':28s} {'tgt_w':>6s} {'price':>9s} "
           f"{'tgt_$':>9s} {'cur_$':>9s} {'delta_$':>9s} {'action':>6s}")
    print(hdr)
    print("-" * len(hdr))

    orders = []
    # trade every name with a target weight (incl. cross-sectional picks outside
    # the base universe) plus anything currently held
    trade_set = sorted(set(t for t, w in weights.items() if abs(w) > 1e-9) | set(held))
    for t in trade_set:
        w = weights.get(t, 0.0)
        cur_sh = float(held.get(t, 0))
        if t not in last_price:
            if cur_sh != 0:                       # held, no price -> close by qty (cover if short)
                side = "sell" if cur_sh > 0 else "buy"
                print(f"{t:7s} {'(exit, no price)':28s} {0:6.1%} {'-':>9s} "
                      f"{'-':>9s} {'-':>9s} {'exit':>9s} {side.upper():>6s}")
                orders.append({"ticker": t, "side": side, "qty": abs(cur_sh)})
            continue
        px = last_price[t]
        tgt_dollar = w * equity
        cur_dollar = cur_sh * px
        delta_dollar = tgt_dollar - cur_dollar
        sigs = ",".join(detail.get(t, [])) if detail.get(t) else "(flat)"

        if abs(w) < 1e-9 and cur_sh != 0:         # target flat -> close long OR short
            action = "SELL" if cur_sh > 0 else "BUY"
            orders.append({"ticker": t, "side": action.lower(), "qty": abs(cur_sh)})
        elif abs(delta_dollar) >= args.min_order:
            action = "BUY" if delta_dollar > 0 else "SELL"
            if args.whole_shares or w < 0 or cur_sh < 0:   # shorts: whole-share only (no fractional)
                qty = abs(int(delta_dollar / px))
                if qty > 0:
                    orders.append({"ticker": t, "side": action.lower(), "qty": qty})
                else:
                    action = "skip"
            else:
                orders.append({"ticker": t, "side": action.lower(),
                               "notional": round(abs(delta_dollar), 2)})
        else:
            action = "hold"                       # inside the no-trade band
        print(f"{t:7s} {sigs:28s} {w:6.1%} {px:9.2f} {tgt_dollar:9,.0f} "
              f"{cur_dollar:9,.0f} {delta_dollar:9,.0f} {action:>6s}")

    print(f"\n{len(orders)} order(s) to reconcile.")
    if not orders:
        print("Portfolio already aligned -- nothing to do.")
        return
    if not args.live:
        print("DRY-RUN -- no orders sent. Re-run with --live to execute on Alpaca paper.")
        return

    print("Submitting orders...")
    # cancel any stale/pending open orders first so a re-run never double-submits
    if not agent.simulated and agent.client is not None:
        try:
            canceled = agent.client.cancel_orders()
            if canceled:
                print(f"  (canceled {len(canceled)} stale open order(s) first)")
        except Exception as e:
            print(f"  [warn] could not cancel open orders: {e}")
    for o in orders:
        res = agent.run({"payload": {"signal": {**o, "order_type": "market",
                                                "time_in_force": "day"}},
                         "strategy_id": f"daily_{args.book}"})
        status = res.get("fill", {}).get("status", res.get("reason"))
        amt = f"{o['qty']:>7g} sh" if "qty" in o else f"${o['notional']:>8,.0f}"
        print(f"  {o['side'].upper():4s} {amt} {o['ticker']:6s} -> {status}")


if __name__ == "__main__":
    main()
