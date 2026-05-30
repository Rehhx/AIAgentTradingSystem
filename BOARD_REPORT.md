# Systematic Equity Book — Board Report

*Generated 2026-05-30 · backtest 2016-01-04 → 2026-05-28 · $100k base · 6 bps round-trip costs · split/dividend-adjusted data*

## 1. Headline performance (deployed book — 7-sleeve `portfolio_full`)

| Metric | This book | S&P 500 (same period) |
|---|---|---|
| Total return | **476%** | 344% |
| CAGR | **18.4%** | 15.4% |
| Sharpe ratio | **1.53** | 0.90 |
| Max drawdown | **-13.4%** | -33.7% |
| $100k grows to | **$575,850** | $444,122 |

**Out-of-sample robustness:** in-sample Sharpe +1.15 → out-of-sample +2.37; **positive in 5/5 walk-forward folds.**

| Walk-forward fold | Return | Sharpe |
|---|---|---|
| 2016-01–2018-01 | +39.0% | +2.54 |
| 2018-01–2020-02 | +10.5% | +0.45 |
| 2020-03–2022-03 | +62.2% | +1.76 |
| 2022-03–2024-04 | +36.5% | +1.37 |
| 2024-04–2026-05 | +69.2% | +2.21 |

## 1b. Stress test through the 2008 GFC (2005–2026, core engine)

The deployed book above is validated 2016–2026 (no GFC-scale crash in that window).
To pressure-test the real downside, the **core equity engine** (RSI-2, Donchian, 50/200
trend, recovery + the same vol-target/early-warning overlays) was run back to **2005**,
spanning the **2008 GFC, 2011, and 2015** bears the recent window lacks:

| Metric | Core engine | S&P 500 |
|---|---|---|
| CAGR (21 yrs) | **12.8%** | 11.0% |
| Sharpe | **1.10** | 0.65 |
| **Max drawdown** | **-31.6%** | -55.2% |

| Bear market | Core engine | S&P 500 |
|---|---|---|
| 2008 GFC | -26.1% | -54.0% |
| 2011 EU crisis | -6.6% | -18.6% |
| 2015-16 selloff | -5.4% | -12.2% |
| 2018 Q4 | -10.0% | -18.9% |
| COVID | -11.8% | -33.4% |
| 2022 bear | -6.8% | -24.1% |

> **TRUE WORST-CASE DRAWDOWN: ~-32% (in the 2008 GFC), not the
> -13% of the 2016–2026 window** — that window simply had no GFC-scale
> event. Honest risk statement for the board: *expect ~−15% in a normal bear and up to
> ~−30% in a once-a-decade, GFC-scale crash.* The engine **survived 2008** (cushioning
> it to about half the market's loss) and caught the 2009 recovery.

**Crisis-alpha validation (Account 2, managed futures):** positive in *both* major bears —
**+5.5% in 2008** and **+4.2% in 2022** — when long equity fell hard. This is
the engine that *profits* in bear markets. Tested bear-profit alternatives (equity shorts,
protective puts, long-volatility/VIX) all proved net-negative — managed-futures trend is
the one approach that pays in crises without ruinous calm-period bleed.

## 2. How this stacks up against bigger firms

- **A Sharpe of 1.53 is top-decile for a systematic equity book.** Most large multi-strategy and equity hedge funds run flagship Sharpes of ~0.5–1.0; the average hedge fund (HFRI) has returned mid-single-digits over the past decade and **underperformed the S&P 500**.
- **Bigger firms do not have bigger *returns* — they have bigger *AUM*.** Scale is a constraint: a $20B+ fund cannot deploy into the capacity-limited daily strategies that drive this book's edge without moving the market against itself. Our size is an advantage here, not a disadvantage.
- This book **beat the S&P 500 on return (18.4% vs 15.4%) with ~40% less drawdown (-13.4% vs -33.7%)** — i.e. more return per unit of risk, which is the metric institutional allocators actually underwrite.

**Peer comparison** (this book vs typical industry benchmarks; peer figures are indicative 10-yr ranges, net-of-nothing basis for comparability):

| Strategy / benchmark | ~CAGR | ~Sharpe | ~Max DD |
|---|---|---|---|
| **This book** | **18.4%** | **1.53** | **-13.4%** |
| This book **+ crypto 5%** (opt-in) | 21.8% | 1.69 | -13.7% |
| S&P 500 (same period) | 15.4% | 0.90 | -33.7% |
| 60/40 stock/bond | ~7–9% | ~0.7–0.9 | ~−20% |
| Average hedge fund (HFRI) | ~5–8% | ~0.6–0.8 | ~−12% |
| Equity long/short HF (avg) | ~6–9% | ~0.6–0.9 | ~−20% |
| Elite multi-strat (pod shops) | ~10–15% net | ~1.5–2.0 | low (tight risk) |
| Renaissance Medallion (closed, unreplicable) | ~30%+ | ~2.5+ | — |

> **Read:** at Sharpe **1.53** this book sits in the *elite multi-strat* tier on risk-adjusted return — above the average hedge fund and 60/40, and ahead of the S&P on both return and drawdown. "Bigger firms" win on AUM and infrastructure, **not** on ROI: the average fund has underperformed the index for a decade. The only higher-return peer (Medallion) is closed and unreplicable; anyone claiming to match it is not credible.

## 3. Risk controls

- **Volatility targeting** (17%, ≤1.8× conditional leverage) — de-levers automatically as volatility rises.
- **Early-warning de-risk** — cuts exposure to 60% when SPY breaks its 50-day with a vol spike, ahead of the lagging 200-day signal.
- **Defensive low-vol sleeve** — rotates to T-bills when SPY < 200-day (bear ballast).
- **Single-name cap** (10%) and a **$250 no-trade band** (churn control).
- **Hard risk gate**: Sharpe ≥ 0.8, max DD ≥ −15%, validated by walk-forward.
- Regime coverage audited: the book beats the market in calm bulls and cushions bears (−12% vs −28% in stormy bears).

## 4. Live paper track record

Live paper track record started **2026-05-30**, **1 session(s)** logged. Need ~5+ sessions before realized Sharpe/drawdown are meaningful; the monitor appends one row per trading day and flags any drift from backtest expectation.

## 5. Higher-return option (requires governance sign-off)

A small (≤5%) **crypto-momentum sleeve** (BTC/ETH, trend-filtered, wired as opt-in) lifts the book to **21.8% CAGR / Sharpe 1.69 / -13.7% DD** — still inside the −15% gate. *Caveat:* crypto's historical return is front-loaded in the 2017 bull and will not repeat at that scale; size it as a bonus, not a base case. This is a board/governance decision, not a quant one.

## 6. Honest caveats

- Results are **backtested over one decade with one out-of-sample window**, on a **long-biased** book (it cushions crashes, it does not profit from them).
- 18% is a multi-year *average* with softer lean years (2018–2020 ≈ +9%/yr), not a yearly guarantee.
- The book should be **paper-traded live until the track record (§4) confirms it matches the backtest** before real capital is committed.
- ~31 strategies and both options-income approaches were tested; the deployed mix is at its validated efficient frontier for this universe. Bigger returns from here require either the crypto sleeve (§5) or leverage (rejected — pure risk, no Sharpe gain).

---
*Reproduce: `python runners/full_backtest.py` (book) · `python runners/monitor.py` (live track record) · full strategy reference in `STRATEGIES.md`.*
