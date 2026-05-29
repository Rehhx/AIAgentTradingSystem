# Board Proposal — Managed-Futures (CTA) Diversification Program

**Prepared:** 2026-05-29
**Decision requested:** approve / decline funding a separate managed-futures book as a diversifier to the equity strategy.

---

## 1. The problem it solves
Our deployed equity book (`portfolio_div`) is excellent on its own terms — ~17% CAGR, −13% max drawdown, positive in 5/5 walk-forward folds — but it is **long-biased equity**. It therefore has **lean years in choppy/sideways markets** (2018–2020 ≈ +2–3%/yr) and is, by construction, correlated to equities. The only structurally different way to earn **positive return in equity-flat or bear years** is a **genuinely uncorrelated** return stream. Managed futures (CTA / trend-following) is the textbook candidate — it profited in 2008 and 2022 when equities fell.

## 2. What it is
A diversified **long *and* short** trend-following program across asset classes — equity indices, government bonds/rates, commodities, and currencies — sized by volatility (equal risk per instrument). Shorting is the key feature: it lets the book **make money when assets fall**, which is what produces "crisis alpha." This is the strategy run by AQR, Man AHL, Winton, etc.

## 3. Prototype evidence (ETF proxies — honest)
We built a proxy version using liquid ETFs (we lack true futures data) — long/short 12-month time-series momentum across 10 asset-class ETFs, vol-targeted:

| Metric | Result |
|---|---|
| Sharpe | 0.52 |
| CAGR | 4.6% |
| Max drawdown | −21.6% |
| **2022 (stock + bond bear)** | **+5.2%** ✅ crisis alpha worked |
| **2018 (Q4 crash)** | **−12.4%** ✗ trend too slow for the fast crash |
| Correlation to equity book | **0.41** (lowest of any diversifier we've tested) |
| Added to portfolio (30%) | Sharpe 1.39→1.32, CAGR 16.2%→12.7% — **dilutive** |

**Honest read:** the proxy is weak standalone, *did* deliver in 2022, but **lost in 2018** and **does not improve the equity book** over 2018–2020. It is real diversification (corr 0.41) but low return — consistent with the well-documented **CTA "trend drought" of ~2011–2020.**

## 4. Industry context (2018–2020)
Managed futures (SG CTA Index) returned roughly **−5.8% (2018), +6.2% (2019), +1.9% (2020)** — about **flat cumulatively.** In other words, **even the professional CTA industry did not profit in this specific window.** This validates that 2018–2020 was structurally hard for diversifiers, not a gap unique to us.

## 5. What a full build requires
| Component | Requirement | Note |
|---|---|---|
| Data | Continuous futures contracts (~25–40 instruments, roll-adjusted) | **Paid** — Norgate (~$30/mo), CSI, or Databento. yfinance futures are unreliable. |
| Broker | Futures account (Interactive Brokers) | Alpaca does **not** offer futures. |
| Signals | Multi-speed trend + carry, long/short, vol-targeted (~10–20% vol) | Reuses our existing trend/vol-target code. |
| Execution | New order path (futures contracts, margin, rolls) | Separate from the equity rebalancer. |
| Effort | **Weeks** — new data feed + broker + execution; a separate book, not a sleeve | |

## 6. Honest expected payoff & risks
- **Upside:** ~0 long-run correlation to equities; genuine crisis alpha (2008, 2022); smooths the combined equity+CTA portfolio's drawdowns.
- **Reality:** **modest standalone Sharpe** (CTAs underperformed 2011–2020); it can *bleed* in calm trending-equity years (2017, 2023–24) and, as shown, **was flat-to-down in 2018–2020 specifically** — so it would **not reliably fix that window.**
- **Risks:** shorting (squeezes, borrow), data/roll complexity, leverage inherent in futures, and ongoing data + IBKR costs.

## 7. Recommendation
Managed futures is the *correct* long-term diversifier and worth pursuing **if the goal is portfolio-level drawdown reduction and crisis insurance** — not as a fix for any single window. It is a **separate program** (paid futures data + IBKR), a multi-week project, with a modest-Sharpe / high-diversification profile. 

**Suggested path:** keep the equity book live (it stands on its own), and greenlight a **time-boxed spike** — license one month of futures data, build the full long/short CTA on ~25 instruments, and judge it on its *combined-portfolio* contribution (Sharpe and drawdown of equity + CTA), not standalone return. Decline if the appetite is for return rather than insurance — the ETF-proxy evidence says the standalone return is low.
