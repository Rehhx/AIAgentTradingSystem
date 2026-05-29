"""
runners/daily_probe.py
----------------------
Quick, honest probe of LOWER-FREQUENCY strategies that the existing 1-minute
search space never touched. The thesis from LESSONS.md + champions.json is that
everything dies from cost drag at 6bps round-trip on 1m bars. Daily / multi-day
holds amortize that cost over a much larger move, so the economics are different.

This is a STANDALONE evidence-gathering script. It does not touch the engine.
Costs: 6 bps charged per round-trip trade (matches the project's stated basis).
Sharpe: daily returns, annualized by sqrt(252). 70/30 train/test split reported
separately so we can see overfitting immediately.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from data.loader import load_ticker

RT_COST = 0.0006          # 6 bps round-trip
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GLD", "JPM"]


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "open":  df["open"].resample("1D").first(),
        "high":  df["high"].resample("1D").max(),
        "low":   df["low"].resample("1D").min(),
        "close": df["close"].resample("1D").last(),
        "vol":   df["volume"].resample("1D").sum(),
    }).dropna()
    return d


def metrics(daily_ret: pd.Series, n_trades: int) -> dict:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 20 or daily_ret.std() == 0:
        return dict(sharpe=0.0, cagr=0.0, maxdd=0.0, n=n_trades, days=len(daily_ret))
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    eq = (1 + daily_ret).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(daily_ret)) - 1
    dd = (eq / eq.cummax() - 1).min()
    return dict(sharpe=round(sharpe, 3), cagr=round(cagr, 4),
                maxdd=round(dd, 4), n=n_trades, days=len(daily_ret))


def backtest_position(close: pd.Series, pos: pd.Series) -> pd.Series:
    """pos in {-1,0,1} held that day (already shifted to avoid lookahead).
    Returns net daily return series after charging RT_COST on each change."""
    pos = pos.shift(1).fillna(0)                       # enter next day -> no lookahead
    ret = close.pct_change().fillna(0)
    gross = pos * ret
    turns = pos.diff().abs().fillna(0)                 # 1.0 per side change
    cost = turns * (RT_COST / 2)                       # half-cost per side
    n_trades = int((pos.diff().abs() > 0).sum())
    return (gross - cost), n_trades


# ---- strategy library (daily) -------------------------------------------------

def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def strat_rsi2_meanrev(d):
    """Connors RSI(2): long when RSI(2)<10 and above 200SMA; exit RSI(2)>70."""
    c = d["close"]
    r = rsi(c, 2)
    sma200 = c.rolling(200).mean()
    pos = pd.Series(0.0, index=c.index)
    long = (r < 10) & (c > sma200)
    exit_ = r > 70
    state = 0
    out = []
    for i in range(len(c)):
        if state == 0 and bool(long.iloc[i]):
            state = 1
        elif state == 1 and bool(exit_.iloc[i]):
            state = 0
        out.append(state)
    return pd.Series(out, index=c.index, dtype=float)


def strat_tsmom_50_200(d):
    """Classic trend: long when 50SMA>200SMA, else flat."""
    c = d["close"]
    return (c.rolling(50).mean() > c.rolling(200).mean()).astype(float)


def strat_donchian_breakout(d):
    """20-day high breakout long, exit on 10-day low (Turtle-lite)."""
    c = d["close"]
    hi = c.rolling(20).max()
    lo = c.rolling(10).min()
    state, out = 0, []
    for i in range(len(c)):
        if state == 0 and c.iloc[i] >= hi.iloc[i]:
            state = 1
        elif state == 1 and c.iloc[i] <= lo.iloc[i]:
            state = 0
        out.append(state)
    return pd.Series(out, index=c.index, dtype=float)


def strat_overnight(d):
    """Hold from close to next open (well-documented index drift)."""
    # special-cased return below; position is 'always long overnight'
    return pd.Series(1.0, index=d["close"].index)


STRATS = {
    "rsi2_meanrev":      strat_rsi2_meanrev,
    "tsmom_50_200":      strat_tsmom_50_200,
    "donchian_breakout": strat_donchian_breakout,
}


def run():
    print(f"{'strategy':22s} {'ticker':6s} | {'train_SR':>8s} {'test_SR':>8s} "
          f"{'full_SR':>8s} {'CAGR':>8s} {'maxDD':>7s} {'trades':>7s}")
    print("-" * 90)
    agg = {}
    for name, fn in STRATS.items():
        srs = []
        for t in TICKERS:
            try:
                d = to_daily(load_ticker(t))
            except Exception as e:
                continue
            pos = fn(d)
            net, n = backtest_position(d["close"], pos)
            split = int(len(net) * 0.7)
            m_tr = metrics(net.iloc[:split], n)
            m_te = metrics(net.iloc[split:], n)
            m_full = metrics(net, n)
            srs.append(m_te["sharpe"])
            print(f"{name:22s} {t:6s} | {m_tr['sharpe']:8.3f} {m_te['sharpe']:8.3f} "
                  f"{m_full['sharpe']:8.3f} {m_full['cagr']:8.2%} "
                  f"{m_full['maxdd']:7.1%} {n:7d}")
        agg[name] = np.mean(srs) if srs else 0.0
        print("-" * 90)

    # overnight effect (close->next open), buy-and-hold benchmark
    print("\n== overnight close->open hold (long index, costs charged daily entry/exit) ==")
    for t in ["SPY", "QQQ"]:
        d = to_daily(load_ticker(t))
        on_ret = d["open"].shift(-1) / d["close"] - 1     # tonight's close -> tmrw open
        on_ret = on_ret - RT_COST                          # charged every night
        intra = d["close"] / d["open"] - 1
        bh = d["close"].pct_change()
        print(f"  {t}: overnight {metrics(on_ret.dropna(),len(on_ret))} ")
        print(f"  {t}: intraday  {metrics(intra.dropna(),len(intra))}")
        print(f"  {t}: buy&hold  {metrics(bh.dropna(),0)}")

    print("\n== mean test-Sharpe across tickers ==")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {k:22s} {v:+.3f}")


if __name__ == "__main__":
    run()
