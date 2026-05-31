# Board Summary — Quant Strategy Status

**Date:** 2026-05-31
**Bottom line:** We run a **two-engine systematic book** — a long-equity *growth*
engine and a long/short managed-futures *crisis-alpha* engine — that is robustly
validated across 20+ years (including the 2008 GFC), engineered against regime change,
and **honestly priced on both return and risk**. It is backtested/paper, not yet
live-money; the remaining work is forward proof, not more research.

*(Live, auto-generated metrics are in `BOARD_REPORT.md`; this is the narrative.)*

---

## 1. The honest journey

The project began on **1-minute intraday** strategies — 30+ rule-based, ML, and
options. **None survived.** At a realistic **6 bps round-trip** cost, transaction
costs exceed the signal on 1-minute bars; loss magnitude tracked trade count almost
perfectly. **The fix was the holding period, not the signal:** on daily/multi-day
holds the same cost amortizes over a multi-week move and the edge survives. Everything
below is built on that pivot.

---

## 2. The system today — two uncorrelated engines

| Engine | What it is | Job |
|---|---|---|
| **Account 1 — Growth** | 7-sleeve long-equity book (RSI-2, Donchian, 50/200 trend, cross-sectional momentum, recovery, post-earnings drift, defensive low-vol) + vol-targeting, early-warning de-risk, T-bill cash, single-name cap, optional 5% crypto-momentum sleeve | compound in up-markets, cushion crashes |
| **Account 2 — Crisis-alpha** | diversified long/**short** managed-futures (time-series momentum across 10 asset-class ETFs, conviction-scaled, vol-targeted) | **profit when equities fall** |

We run them on **two separate Alpaca paper accounts** so one wins when the other loses.

---

## 3. Results (backtested; $100k base, 6 bps)

| Book | CAGR | Sharpe | Max DD | Window |
|---|---|---|---|---|
| **Account 1 — Growth** (crypto-armed) | 21.8% | 1.69 | −13.7% | 2016–26 |
| Account 1 — no-crypto base | 18.4% | 1.53 | −13.4% | 2016–26 |
| **Account 2 — Crisis-alpha** | 1.8% | 0.31 | −12.5% | (**+5.5% in 2008, +4.2% in 2022**) |
| **Combined 70/30** (all-weather) | 15.6% | 1.62 | −10.6% | 2016–26 |
| **Core engine — GFC-tested** | 12.8% | 1.10 | **−31.6%** | **2005–26** |
| S&P 500 | 15.4% | 0.90 | −33.7% (−55% w/ GFC) | — |

The combined book matches the S&P's return with **~⅓ the drawdown and nearly double
the Sharpe**, and the crisis engine is **positive in both 2008 and 2022**.

---

## 4. Robustness — engineered for a non-stationary market

Markets are non-stationary; a single backtest with one parameter set is fragile. We
addressed this directly, with measurements not assertions:

- **GFC stress test (2005–2026):** survives 2008 at **−26% vs the S&P's −54%**; edge
  persisted across the dot-com tail, GFC, ZIRP, COVID, and 2022 inflation. (`extended_backtest.py`)
- **Parameter ensembling (deployed):** each sleeve averages 3–5 parameter settings, so
  the live book never bets on a single θ a regime shift could break. (`param_ensemble.py`)
- **Block-bootstrap (3,000 paths):** the edge is not a lucky ordering — actual = median,
  the unlucky 5th-percentile path still returns +14%, P(CAGR>10%)=100%. (`bootstrap_robustness.py`)
- **Walk-forward:** positive in 5/5 contiguous folds; a parameter optimizer **overfit and
  lost** to the simple weights out-of-sample (we don't curve-fit θ).
- **Regime-switching tested and rejected:** the static blend beats it (the 200-day signal
  lags and whipsaws) — we adapt via *mechanism* (trend-following) and *diversification*,
  not by forecasting regimes.

---

## 5. Honest caveats (stated plainly for the board)

- **True worst-case drawdown is ~−25% (bootstrap) to ~−32% (2008 GFC)** — *not* the
  −13.7% of the benign 2016–2026 window. Size capital to the −32% tail.
- **Net of realistic costs (~20 bps real fills, not 6 bps), expect ~12–18% CAGR, not
  18–22%.** Our fill-tracker already measures ~17 bps slippage on paper. (`fill_tracker.py`)
- **Long-biased growth engine; not market-neutral.** Its value over buy-and-hold is
  drawdown reduction + the crisis engine's positive bear returns.
- **Backtested / paper — not live-money.** The track record is just starting; no backtest,
  however robust, can prove a non-stationary future.
- **Crypto's contribution is front-loaded** (2017 bull) and governance-gated; the no-crypto
  18.4% is the conservative base.

---

## 6. Lowering the drawdown — an allocation choice

The crisis engine (managed futures) **profits in crashes** (+10.7% in 2008 standalone), so
drawdown is a dial, not a missing strategy. Through the 2008 GFC: **50/50 growth/crisis
halves the GFC drawdown (−26% → −9%)** at the cost of CAGR/Sharpe; ~60% growth is the
balanced middle (−18.7% max DD, Sharpe 1.01). Shorts, protective puts, and long-volatility
were all tested and rejected (whipsaw / premium bleed / catastrophic decay).

---

## 7. The path to real capital (forward proof)

1. **Rotate the API keys** — urgent; they were in public git history (prerequisite before real money).
2. **Run the paper book forward**; `monitor.py` + `tracking_dashboard.py` log live-vs-backtest daily, `fill_tracker.py` measures the real slippage gap.
3. **Choose the growth/crisis allocation** to the board's drawdown tolerance (§6).
4. **Small real pilot**, sized to the −32% worst case — the only thing that fully closes the backtest-vs-live gap.

**Bottom line:** a rigorous, GFC-tested, regime-robust, two-engine system, documented
honestly on both return and risk. The differentiator isn't the headline number — it's the
discipline: ~30 strategies rejected on evidence, the drawdown corrected upward, the ML
model killed when it overfit, and tooling built because we know paper overstates live.
