# Résumé material — Quant-Agent Trading System

Honest, metric-driven copy for a résumé / LinkedIn / portfolio. **All figures are
backtested (2016–2026 deployed window; 2005–2026 for the GFC stress test) or
paper-trading** — labeled as such so they hold up under interview scrutiny. The
system is a **two-engine book**: a long-equity *growth* engine + a long/short
managed-futures *crisis-alpha* engine, each with risk overlays and forward monitoring.

---

## Project header

> **Two-Engine Systematic Trading System (Equity + Managed Futures)** — *Python, pandas/NumPy, scikit-learn, Alpaca, Claude Agent SDK*

---

## Full bullets (pick 5–6)

- Built an **end-to-end systematic trading system** — research → backtest → risk gate
  → live (paper) execution → monitoring → forward validation — across two uncorrelated
  engines: a 7-sleeve **equity growth** book (mean-reversion, breakout, trend, cross-
  sectional momentum, recovery, post-earnings drift, defensive low-vol) and a
  diversified **long/short managed-futures** book (time-series momentum across 10 asset
  classes).
- Backtested the growth book to **Sharpe 1.53–1.69 / 18–22% CAGR / −13% drawdown**
  (2016–2026) and **stress-tested the full system back to 2005 through the 2008 GFC**,
  where it lost only −26% vs the S&P's −54% and the managed-futures engine was **+10.7%**.
- Engineered for **non-stationarity / regime change**: **parameter ensembling** (each
  sleeve averages 3–5 settings rather than betting on one θ), **diversification across
  regime-winners** instead of regime-timing (empirically beat a regime-switching model),
  and **walk-forward + block-bootstrap** validation (3,000 resampled paths; edge positive
  in 100% of them).
- Established the project's core finding — **1-minute intraday strategies don't survive
  realistic transaction costs across 30+ tested strategies** — and pivoted to daily/
  multi-day holds where the edge persists; quantified the live-vs-backtest gap with a
  **fill-quality tracker** (real fills ~17 bps vs the 6 bps backtest assumption).
- Applied **machine learning** (gradient-boosting + logistic) to predict trade success
  and rank cross-sectional returns; **rigorously found it did not beat the rule-based
  signals out-of-sample** (AUC ≈ 0.50) — a disciplined negative result that avoided
  overfitting low-signal financial data.
- Implemented the full **risk framework**: volatility targeting, regime-based de-risk
  overlays, single-name concentration caps, idle-cash → T-bills, and an honest
  worst-case drawdown of **~−32% (2008)** — corrected up from the benign-window −13%.
- Built **live monitoring + a paper-vs-backtest tracking dashboard** that flags drift,
  drawdown breaches, and regime-posture mismatches day-by-day (Alpaca paper API).

---

## Tight 3-bullet version (space-constrained)

- Built a two-engine systematic trading system (equity growth + managed-futures
  crisis-alpha) end-to-end: research, backtest, risk, execution, and live monitoring.
- Backtested to **Sharpe ~1.5–1.7**, **GFC-stress-tested back to 2005** (−26% in 2008
  vs the market's −54%), and validated with walk-forward, **block-bootstrap (3,000
  paths)**, and **parameter ensembling** to guard against regime change / overfitting.
- Quantified the backtest-vs-live gap with a **fill-quality tracker** and reported the
  honest cost-adjusted return and **~−32% true worst-case drawdown** — not a flattering
  paper number.

---

## Skills line

> Python · pandas · NumPy · scikit-learn · quantitative backtesting · walk-forward &
> bootstrap validation · regime analysis · risk management (vol-targeting, drawdown
> control) · managed futures / trend-following · transaction-cost modeling · REST APIs
> (Alpaca, Finnhub, yfinance) · Git

---

## Your 3 best interview talking points (lead with these)

1. **Killing your own idea on evidence:** *"My core finding was that 1-minute intraday
   strategies don't survive transaction costs — I proved it across 30+ strategies, then
   pivoted to daily holds where the edge persists."*
2. **Engineering for non-stationarity** (the deepest quant problem): *"I don't bet on a
   single parameter or try to time regimes — I use parameter ensembling, diversify across
   strategies that each win in different regimes, and validate across the dot-com bust,
   the GFC, COVID, and 2022 inflation."*
3. **Backtest-vs-live realism** (what actually gets you hired): *"Paper backtests assume
   perfect liquidity, so I built a fill-quality tracker — real fills cost ~17 bps vs the
   6 bps assumed — and I report the cost-adjusted return and the true −32% GFC drawdown,
   not the flattering figure."*

---

## Honest framing (do not skip)

State results as **backtested / paper research**, never "I made X% with real money."
The intellectual honesty *is* the differentiator — you rejected ~30 strategies on
evidence, corrected your own drawdown, killed the ML model when it overfit, and built
tooling specifically because you knew paper overstates live. An interviewer who catches
an inflated claim discards the whole résumé; one who sees the rigor and candor remembers
it. This reads as a *junior-quant-with-senior-instincts* project.
