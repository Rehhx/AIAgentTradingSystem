# Résumé material — Quant-Agent Trading System

Honest, metric-driven copy for a résumé / LinkedIn / portfolio. All figures are
**backtested (2016–2026) or paper-trading** results — labeled as such so they
hold up under interview scrutiny. Headline book: 6-sleeve `portfolio_full`,
vol-targeted (17%, ≤1.8× conditional leverage).

---

## Project header

> **Autonomous Multi-Agent Quantitative Trading System** — *Python, Claude Agent SDK, Alpaca, pandas/NumPy*

---

## Full bullets (pick 4–5)

- Designed and backtested a **six-strategy daily-rebalanced equity ensemble**
  (mean-reversion, breakout, trend, cross-sectional momentum, recovery-thrust,
  post-earnings drift) achieving a **1.46 Sharpe, 18.2% annualized return, and
  −13.1% max drawdown** over a 2016–2026 backtest — **positive in all 5
  walk-forward folds**.
- Engineered the full validation stack — **anchored walk-forward analysis with
  per-fold parameter re-optimization, volatility targeting, risk-parity
  weighting, and a regime-based early-warning de-risk overlay** — gated by a
  quantitative risk screen (Sharpe / drawdown / win-rate / trade-count) before
  any strategy could deploy.
- Diagnosed and fixed a **data-integrity bug** in which unadjusted
  split/dividend prices were corrupting backtested fills; cross-validated
  against an independent source, **lifting the affected book's Sharpe from ~1.09
  to ~1.32**.
- Built a **live paper-trading rebalancer** on the Alpaca API with
  target-vs-current order reconciliation, fractional/notional sizing, no-trade
  bands, and stale-order cancellation; automated daily pre-market execution via
  Task Scheduler.
- Architected a **multi-agent research pipeline** (research → code generation →
  backtest → risk gate) on the Claude Agent SDK, with an automated
  validation-retry loop and a persistent ledger of every strategy tested.
- Established a key research finding: **systematically showed that 1-minute
  intraday strategies fail to survive realistic 6 bps transaction costs across
  30+ tested strategies**, then pivoted the system to daily/multi-day holds
  where the edge persists.
- Researched **no-leverage options income strategies** (cash-secured put-write,
  covered call) that harvest the volatility risk premium, modeling premiums via
  Black-Scholes and stress-testing the implied-vs-realized-vol assumption.

---

## Tight 3-bullet version (space-constrained)

- Built a Python multi-agent quant system that researches, validates, and
  paper-trades daily equity strategies end-to-end.
- Backtested a 6-strategy vol-targeted ensemble to **Sharpe 1.46 / 18.2% CAGR /
  −13.1% max DD**, validated across 5 walk-forward folds with a strict risk gate.
- Automated live paper execution (Alpaca) with order reconciliation and
  scheduled daily rebalancing; caught and fixed a price-adjustment data bug that
  had been corrupting results.

---

## Skills line

> Python · pandas · NumPy · quantitative backtesting · walk-forward validation ·
> risk management · time-series analysis · REST APIs (Alpaca, yfinance) ·
> multi-agent LLM orchestration (Claude Agent SDK) · Git

---

## Interview tip

Lead with the **"intraday doesn't survive costs → daily does"** finding. It's
the most senior-sounding point here: it shows you can kill your own idea on
evidence and reason about transaction-cost economics — exactly what quant/eng
interviewers probe for. Be ready to explain *why* (6 bps round-trip amortizes
over a multi-week daily hold but dominates a 1-minute signal).

> **Honesty note:** these are backtested / paper-trading results over one decade
> and one out-of-sample window, on a long-biased equity book — not live-money,
> not market-neutral. State that plainly if asked; it reads as rigor, not weakness.
