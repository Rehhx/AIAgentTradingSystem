# Resume Build Plan — Quant Research/Dev Portfolio

The goal of this plan is **not** more trading alpha (the honest ceiling is ~1.59 Sharpe).
It is to make the rigor already demonstrated in this project **legible to a quant firm**
(D.E. Shaw and peers). Trading returns don't impress them; methodology and intellectual
honesty do. Each tier below builds something that signals exactly that.

---

## Recommended sequence (dependencies matter)

```
Tier 1A  stat-rigor library  ──┐  (keystone: makes everything else credible)
Tier 1B  research write-up   ──┘  consumes 1A output
Tier 2A  event-driven backtester  ──┐  (foundation for ML)
Tier 3   ML, time-aware validation ─┘  runs on 2A, validated by 1A
Tier 2B  Rust order book  (isolated showpiece — slot into any long block)
```

Do **1A first** — highest signal-per-hour, extends the existing
`_metrics_from_returns` / `walk_forward_folds`, and every later number gets reported
through it.

---

## TIER 1A — Statistical-rigor library  ⭐ keystone   [STATUS: DONE — 2026-06-04]

**RESULT (13 sleeves, 2016-2026, 2615 days):**
- Best sleeve `abs_momentum` naive Sharpe 1.20 -> **Deflated Sharpe 99.4% (PASS** vs a
  0.39 selection hurdle): the individual edge is real, not luck-of-13-trials.
- **PBO = 52%**: selecting the in-sample-best sleeve has ~zero OOS predictive power —
  the sleeves are near-interchangeable, so betting on "the best backtest" is overfitting.
- **White's RC p=0.83 / Hansen SPA p=0.83**: no sleeve beats buy-&-hold SPY on raw
  return after snooping correction. The sleeves' value is risk reduction, not alpha.
- **Conclusion:** the rigor battery *validates the architecture* — an equal-weight,
  vol-targeted ENSEMBLE with a crash sentinel is the correct response to PBO=52%, not a
  bet on a single backtested champion. This is the narrative for RESEARCH.md (1B).

**Goal:** correct the Sharpe for the fact that ~10 sleeves were tried, and quantify overfitting.

**Build:**
- `analytics/significance.py` — `deflated_sharpe_ratio()`, `expected_max_sharpe(n_trials)`,
  `min_track_record_length()` (Bailey & López de Prado)
- `analytics/pbo.py` — `cscv_pbo()` (Combinatorially Symmetric Cross-Validation →
  probability of backtest overfitting)
- `analytics/reality_check.py` — `whites_reality_check()`, `hansen_spa()` (bootstrap
  data-snooping tests)
- `runners/rigor_report.py` — runs all of it against the 10 sleeves, prints the honest
  table: *N trials, naive best Sharpe, deflated Sharpe, PBO%*
- `tests/test_significance.py`, `tests/test_pbo.py` — analytical known-answer cases

**Resume bullet:** "Computed Deflated Sharpe Ratio and Probability of Backtest Overfitting
across 10 candidate strategies; reported the deflated 1.x rather than the inflated nominal
Sharpe."

**Effort:** 1–2 days · **Difficulty:** math-heavy but well-documented formulas
**Done when:** `rigor_report.py` prints deflated Sharpe + PBO for the book and tests pass.

---

## TIER 1B — Research write-up   [STATUS: DONE — 2026-06-04]

**Delivered:** `RESEARCH.md` — abstract, architecture, methodology, the §3 rigor results
(DSR 99.4% / PBO 52% / RC+SPA p=0.83), the skew-artifact and BNY-glitch case studies, the
1.59 ceiling with the diversification math, and a rejected-strategies table.

**Goal:** the communication artifact. Quant firms weight clear writing heavily.

**Build:** `RESEARCH.md` (or a short PDF) — abstract, methodology (walk-forward, costs,
look-ahead controls), three case studies: the **skew artifact** (3× overstatement), the
**BNY glitch** (data-quality controls), the **1.59 ceiling argument**, the **deflated
results from 1A**, and a "what we rejected and why" table (pairs, sector mom, covered calls).

**Resume bullet:** "Authored a research report documenting why five candidate strategies
were rejected, including a volatility-skew pricing artifact that overstated returns 3×."

**Effort:** ½–1 day · **Difficulty:** low (analysis already exists)
**Done when:** a reader who's never seen the code understands what works, what doesn't, why.

---

## TIER 2A — Event-driven backtester (look-ahead-free by construction)   [STATUS: DONE — 2026-06-04]

**Delivered:** `backtest/` package (events, data, strategy, portfolio, execution, engine).
The DataHandler hard-bounds reads to `iloc[:cursor+1]` so look-ahead is impossible by
construction. `runners/bt_parity.py` validates against the vectorized book: trend_5020 on
SPY -> **Sharpe 0.751 = 0.751, correlation 1.000000, total return within 8 bps**. 8 new
engine tests (incl. the no-look-ahead firewall assertion); **52 tests passing total.**

**Goal:** a reusable engine that *cannot* peek ahead — the infrastructure signal.

**Build:** `backtest/` package
- `events.py` — `MarketEvent → SignalEvent → OrderEvent → FillEvent` queue
- `data.py` — point-in-time bar handler (yields one bar at a time; can't see the future)
- `strategy.py` — pluggable strategy interface
- `portfolio.py` — position/cash accounting, mark-to-market
- `execution.py` — explicit slippage + commission model
- `runners/bt_parity.py` — port the **trend sleeve** onto it, prove it matches the
  vectorized version within tolerance
- `tests/test_backtest_engine.py` — fill logic, no-look-ahead assertion, accounting invariants

**Resume bullet:** "Built an event-driven backtesting engine with point-in-time data
handling that prevents look-ahead bias by construction; validated against vectorized results."

**Effort:** 3–5 days · **Difficulty:** moderate (design discipline > algorithms)
**Done when:** ported sleeve matches vectorized Sharpe to 2 decimals and no-look-ahead test passes.

---

## TIER 3 — ML with time-aware validation

**Goal:** ML done *correctly* — the methodology is the value, even though it likely won't
beat the simple book (state that honestly).

**Build:** `ml/` package
- `cv.py` — `PurgedKFold` + embargo (López de Prado) to kill leakage
- `labels.py` — triple-barrier labeling
- `runners/ml_signal.py` — gradient-boosting model on sleeve features, run on the **2A
  engine**, evaluated through **1A's deflated Sharpe**; report feature importance (MDA)
- `tests/test_purged_cv.py` — assert no train/test temporal overlap after purge+embargo

**Resume bullet:** "Applied purged & embargoed cross-validation and triple-barrier
labeling; demonstrated that a gradient-boosted signal did not survive deflated-Sharpe
testing out-of-sample, and explained why via feature-decay analysis."

**Effort:** 2–3 days · **Difficulty:** moderate
**Done when:** CV provably leak-free and the honest OOS result is reported through 1A.

---

## TIER 2B — Rust limit-order-book + matching engine  ⭐ quant-dev showpiece

**Goal:** the low-latency/systems signal that a pure-Python project lacks.

**Build:** `orderbook-rs/` crate (Rust + PyO3/maturin bindings)
- Price-time-priority matching, add/cancel/modify, market & limit orders, partial fills
- `benches/` — throughput (orders/sec) + latency percentiles (p50/p99)
- Python binding + `runners/ob_demo.py` replaying a synthetic order flow
- Tests in **both** Rust (`cargo test`) and Python

**Resume bullet:** "Implemented a price-time-priority matching engine in Rust with Python
bindings; benchmarked at N orders/sec, p99 latency X µs."

**Effort:** 4–7 days (more if Rust is new) · **Difficulty:** highest, highest quant-dev payoff
**Done when:** matches a reference scenario's fills and benchmarks are published in the README.

---

## Timelines & cuts

| Track | Build | Total |
|---|---|---|
| **Minimum viable** (do these 3) | 1A + 1B + 2A | ~1 week |
| **Researcher-leaning** | + Tier 3 | ~1.5–2 weeks |
| **Dev-leaning** | + Tier 2B | ~2–3 weeks |
| **Everything** | all five | ~3–4 weeks part-time |

**If you only build one thing:** Tier 1A. It's the cheapest and strongest signal — the
difference between "I backtested some strategies" and "I think like a quant researcher."
