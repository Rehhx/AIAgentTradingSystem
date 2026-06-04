# A No-Leverage Multi-Engine Trading System — Research Report

*Last updated: 2026-06-04*

## Abstract

This report documents the research behind a three-engine, no-leverage systematic
trading book and — more importantly — the **statistical discipline used to decide what
is real**. The headline finding is deliberately humbling: across the 13 equity timing
strategies tested, the single best in-sample performer has **~zero out-of-sample
predictive power** (Probability of Backtest Overfitting ≈ 52%), and **no strategy beats
buy-and-hold SPY** on raw return after correcting for data-snooping (White's Reality
Check / Hansen's SPA *p* ≈ 0.83). That result is not a failure — it is the evidence base
for the system's design choice: an equal-weight, volatility-targeted **ensemble** with a
crash sentinel, rather than a bet on a backtested champion. The honest performance
ceiling of the deployed book is **Sharpe ≈ 1.59** (no leverage, max drawdown −7.8%),
and this report explains, with the diversification math, why reaching a *true* 2.0 would
require either leverage (which re-adds tail risk) or a genuinely new uncorrelated source
of return.

---

## 1. System architecture

The system is split into three independent engines ("accounts"), chosen so their return
streams are as uncorrelated as a retail data budget allows:

| Engine | Mandate | Role |
|---|---|---|
| **E1 — Equity book** | 7 long/flat sleeves over a 10-name quality universe + a crash sentinel | the workhorse |
| **E2 — Managed futures** | time-series & cross-sectional momentum on liquid futures | crisis alpha |
| **E3 — Defined-risk options** | deep-ITM LEAPS as share-replacement | leveraged exposure, capped loss |

**Design constraint — no margin, ever (max leverage 1.0).** Volatility targeting scales
exposure *down* in high-vol regimes but is hard-capped at 100% gross, so the book
**cannot be margin-called**. This is a deliberate trade of upside for survivability; as
shown in §6, leverage raises CAGR but not Sharpe, while adding drawdown and tail risk.

---

## 2. Methodology

Every result in this report is produced under the same controls:

- **No look-ahead.** Signals are computed on each day's close and entered the *next* day
  (`signal.shift(1)` before the backtest loop). This single fix corrected historical
  Sharpes that were inflated by same-bar entry.
- **Realistic costs.** 3 bps per side / **6 bps round-trip** charged on every position
  change. On intraday 1-minute bars this cost dominated every signal; on daily holds it
  is amortized over a multi-week move and becomes tractable — which is *why* the project
  trades daily, not intraday.
- **Split-adjusted data.** Daily bars use dividend/split-adjusted yfinance. Switching
  from raw to adjusted data was not cosmetic: it raised the blended book's Sharpe from
  1.09 to 1.32 and made split stocks (GOOGL, AAPL, AMZN, NVDA) safe.
- **Walk-forward, not one split.** Strategies are evaluated over 5 contiguous
  out-of-sample folds; an edge must be positive in most folds, not one lucky window.
- **Selection-bias correction.** All candidate Sharpes are passed through the
  statistical-rigor battery of §3 before any are believed.

---

## 3. Statistical rigor — does the edge survive?

The core principle: *a positive backtested Sharpe is the null hypothesis, not the
finding.* Three independent tests are applied to the full set of 13 sleeves
(2016–2026, 2,615 trading days; `runners/rigor_report.py`, `analytics/`).

### 3.1 Deflated Sharpe Ratio (Bailey & López de Prado)

The naive Sharpe of the best sleeve is inflated by (a) non-normal returns and (b)
selection across many trials. The Deflated Sharpe Ratio corrects both.

| Quantity | Value |
|---|---|
| Best sleeve | `abs_momentum` (6-month time-series momentum) |
| Naive annualized Sharpe | 1.20 |
| Trials searched (N) | 13 |
| E[max Sharpe \| null] — the selection hurdle | 0.39 |
| Probabilistic Sharpe vs 0 | 100.0% |
| **Deflated Sharpe Ratio** | **99.4% (PASS)** |
| Min track-record length | ~1,107 obs (~4.4 yrs) |

**Read:** the best sleeve's positive risk-adjusted return *versus zero* is real — it is
not an artifact of having tried 13 strategies. But "beats zero" is a low bar for a
long-biased equity strategy (equities drift up). The next two tests raise the bar.

### 3.2 Probability of Backtest Overfitting — CSCV

Combinatorially Symmetric Cross-Validation asks: if I pick the best strategy
in-sample, where does it rank out-of-sample?

| Quantity | Value |
|---|---|
| Splits / combinations | 16 / 12,870 |
| **PBO** | **52%** |
| Median OOS-rank logit | +0.00 |

**Read:** a PBO near 50% means selecting the in-sample winner is a *coin flip*
out-of-sample. The 13 sleeves are long-biased variations on the same 10 names — they are
near-interchangeable, so "which backtest is best" carries no generalizable information.
**This is the central justification for the ensemble.**

### 3.3 Data-snooping — White's Reality Check & Hansen's SPA

Both tests use a stationary block bootstrap to ask whether the *best* strategy beats
buy-and-hold SPY after accounting for the number of strategies searched.

| Test | *p*-value |
|---|---|
| White's Reality Check | 0.83 |
| Hansen's SPA | 0.83 |

**Read:** *p* ≈ 0.83 ≫ 0.05 — no sleeve reliably out-*returns* passive SPY. Because the
sleeves sit in cash part of the time, they deliver comparable-or-lower return with
**lower risk**. Their value is drawdown reduction, not alpha. *(Caveat: the Reality Check
compares raw returns, which is unfair to a partially-in-cash strategy; a
Sharpe-differential version is the apples-to-apples follow-up — see §8.)*

### 3.4 Synthesis

The three results compose into one honest conclusion:

> The sleeves earn a genuine positive Sharpe, but you cannot pick the best one ex ante,
> and none beats simply holding the index. The correct response is therefore **not** to
> deploy a champion sleeve — it is to **diversify across them** (equal weight),
> **target volatility**, and **overlay a crash sentinel** to cut the left tail. The
> architecture is the conclusion of the statistics, not a prior.

---

## 4. Case study: the volatility-skew artifact

A covered-call ("buy-write") income overlay initially *looked* excellent — a backtest
showed **Sharpe 1.29 vs 0.76 for buy-hold**, seemingly free risk-adjusted return. It was
an artifact.

- The backtest priced the sold call with Black-Scholes using **VIX** as the implied vol.
- VIX is approximately **at-the-money** implied vol. Equity options carry a **skew**:
  out-of-the-money calls trade *cheaper* than ATM.
- A real-chain snapshot of a 3%-OTM SPY call (~23 DTE) **bid $1.49 versus the
  Black-Scholes mid of $4.44** the backtest assumed — i.e. the model collected **34%** of
  the premium it booked. The true OTM-call IV was ~10%, not VIX's 15.8%.

**Lesson (now codified):** never price a non-ATM option off VIX; the error was ~3×.
Deep-ITM LEAPS (the E3 engine) are ~80% intrinsic, so the IV error barely moves them —
which is *why* the options engine uses share-replacement LEAPS, not OTM premium-selling.
The covered-call strategy was **rejected**.

---

## 5. Case study: data-quality controls

An automated liquidator is only as safe as its price feed. A paper-feed glitch reported
**BNY at −92%** (entry $139.77, stale mark $10.44; the real price was ~$142, the position
was actually up ~2%). A naive stop would have "sold" into a phantom crash.

The stop-guard's **sanity floor** caught it: a drop > 40% versus entry is flagged
**SUSPECT and never auto-sold**. The glitch self-resolved the next session (transient
feed lag). The principle — *implausible prices are data errors until proven otherwise* —
is now a permanent control (`stop_guard.py`).

---

## 6. The diversification ceiling — why 1.59, not 2.0

The deployed E1 book (7 sleeves + crash sentinel, no leverage):

| Config | Sharpe | Max DD | COVID DD | Walk-forward |
|---|---|---|---|---|
| 7-sleeve book | 1.55 | −9.4% | −9.4% | — |
| **+ VIX-term-structure crash sentinel** | **1.59** | **−7.8%** | **−7.7%** | **5/5 folds** |

The sentinel (spot VIX ≥ 3-month VIX = backwardation → de-risk to 60%) is only ~0.69
Sharpe standalone on SPY, but it is a *crash specialist* complementary to the book, so it
lifts the system Sharpe while cutting the tail.

**Why not higher?** Diversification's lift obeys, roughly:

```
combined Sharpe  ≈  signal Sharpe  ×  √(independent bets per year)
```

A daily-rebalanced retail book has a bounded number of *independent* bets per year, and
the available sleeves are mutually correlated (§3.2), so the term under the root saturates
— capping a realistic daily book at ~1.5–1.7. Two things that do **not** break the
ceiling:

- **Leverage.** At 1.0× the book is 10.5% CAGR / 1.55 Sharpe / −9.4% DD; at 1.8× it is
  18.4% CAGR / 1.53 Sharpe / −13.4% DD. Leverage scales return and drawdown together —
  **Sharpe is flat** — and re-introduces margin-call risk the design exists to avoid.
- **Adding the managed-futures engine over 2016–2026.** It is genuinely uncorrelated and
  crisis-positive, but its weak standalone Sharpe in this (largely bull) sample *lowered*
  the blended Sharpe; its value is regime insurance, not headline Sharpe.

A *true* sustained 2.0 requires a genuinely new uncorrelated alpha (scarce at retail data
budgets) or leverage (re-adding tail risk). Claiming 2.0 from this data would be
overfitting — exactly what §3 is built to prevent.

---

## 7. Rejected strategies (and why)

| Strategy | Result | Why rejected |
|---|---|---|
| Covered-call income | fake Sharpe 1.29 | volatility-skew pricing artifact (§4); real premium capture 34% |
| Pairs / stat-arb (10 pairs) | Sharpe ~0.51 | classic pairs alpha appears arbitraged away |
| Sector-momentum rotation | Sharpe ~0.47 | correlated to existing trend sleeves; no diversification |
| `zscore_revert` | Sharpe 0.52 | 0.72 correlated to trend — redundant |
| Naked put-writing | −24% DD | uncompensated tail risk |
| Managed-futures blend (2016–26) | lowers combined Sharpe | crisis insurance, not Sharpe — kept small/separate |

The discipline: a sleeve earns a seat only if it is **competitive after costs *and*
diversifying** (low correlation to what's already deployed). Most candidates fail the
second test, not the first.

---

## 8. Limitations & next steps

- **Reality Check on raw returns** penalizes partially-in-cash strategies; a
  Sharpe-/risk-adjusted differential test is the fairer comparison and is the next
  addition to `analytics/reality_check.py`.
- **Options pricing is approximated** with Black-Scholes on VIX. Real historical IV
  surfaces (the one data purchase with genuine ROI) would both eliminate the §4 class of
  error and open vol-surface strategies — the rare inefficiency still available at this
  scale.
- **Sample regime.** 2016–2026 is bull-heavy; the managed-futures and sentinel engines
  are deliberately retained for the regimes this sample under-weights.

---

## Reproducing the results

```bash
python runners/rigor_report.py        # §3 — DSR, PBO, Reality Check / SPA
python runners/sentinel_book_wf.py    # §6 — book + crash sentinel, walk-forward
python runners/leverage_compare.py    # §6 — Sharpe is flat across leverage
pytest tests/                         # 44 tests incl. the statistical-rigor battery
```

Statistical-rigor library: `analytics/significance.py` (Deflated/Probabilistic Sharpe,
min track-record length), `analytics/pbo.py` (CSCV), `analytics/reality_check.py`
(White's RC, Hansen's SPA). Methodology lessons: `LESSONS.md`.
