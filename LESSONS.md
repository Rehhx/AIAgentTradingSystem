# LESSONS.md

Institutional knowledge from this project — what we tried, what worked, what
didn't, and why. Read this before proposing the same experiment again.

---

## 1. Engine bugs that mattered

Found and fixed during build-out. Their fingerprints in old results files are
no longer present in current ones — but if you see results inconsistent with
these fixes, suspect a regression.

| Bug | Symptom | Fix |
|---|---|---|
| Sharpe annualized by √(252×390) | Reported Sharpes like -56 | Daily-resample equity, annualize by √252 |
| Cash-only equity hid open PnL | Equity flat between trades | Mark-to-market: `capital + shares*(c-entry_px)*pos` |
| Same-bar lookahead | Inflated train Sharpes | `signal = signal.shift(1)` before backtest loop |
| ORB re-entered every bar | Thousands of trades per day | Per-day `traded_long`/`traded_short` flags |
| Strategy lookup matched substrings | `bb_band_touch_revert_v2` ran v1 | Prefer exact match, then substring |
| `noise_area_breakout` reindex tz mismatch | 0 trades on full universe | Use date strings as bucket keys |
| `dump_trades` Δ encoding crash | Windows cp1252 print failed | ASCII-safe print + `encoding="utf-8"` on writes |

---

## 2. Strategies tested — current verdict

All results are from honest walk-forward (70/30 train/test) on 2022-2025 1m
bars at COMMISSION=0.0001 + SLIPPAGE=0.0002 (~6bps round-trip).

### Confirmed dead (do not revisit without new mechanism)

| Strategy | Best train SR | Test SR | Notes |
|---|---|---|---|
| `bb_squeeze` on MSFT | +1.18 | **-2.03** | Full-period +0.26 was overfit |
| `bb_squeeze` on CAT | +0.87 | **-0.43** | Full-period +0.34 was overfit |
| `bb_squeeze` regime-filtered (bull) | +0.47 | -0.30 | 4 trades in test; noise |
| `bb_squeeze` regime-filtered (bear) | +0.47 | 0.00 | 0 trades in test |
| `bb_squeeze` regime-filtered (high_vol) | +0.37 | 0.00 | 0 trades in test |
| `bb_squeeze` regime-filtered (neutral) | +0.18 | -0.73 | Overfit |
| `noise_area_breakout` on AAPL | +1.47 | **-1.68** | Full-period +0.34 was overfit; Zarattini Sharpe 1.33 claim doesn't survive our cost basis |
| `noise_area_breakout` aggregate | +0.58 | -1.00 | Same pattern |
| `qqq_spy_dispersion` | n/a | -22 to -26 | Signal logic exits too quickly on noise; engine is fine |
| `momentum`, `ema_crossover` | n/a | -5 to -8 | Pure noise at 1m on equities |
| `bb_band_touch_revert` (v1) | n/a | -11 to -18 | 49% WR but payoff asymmetry kills it |
| `bb_band_touch_revert_v2` (RSI confluence + breach) | n/a | -3.7 to -5.1 | RSI filter cut Sharpe drag in half; still negative |
| `half_hour_continuation` thr=3bps | n/a | -7.7 | Too many trades, signal too weak |
| `half_hour_continuation` thr=25bps | n/a | -0.56 | Closest to breakeven; signal real but marginal |
| `trend_ride` gated to bull/bear | n/a | -1.58 | Market gate cut drag 2x but didn't flip positive |
| `closing_auction_drift` (autonomous) | n/a | -6.33 | Novel but bleeds |
| `pdh_pdl_liquidity_grab_fade` (autonomous) | n/a | -6.07 | Novel but bleeds |
| `lunch_hour_vwap_anchor_revert` (autonomous) | n/a | **-36.05** | 13,806 trades — cost drag catastrophic |

### Hindsight patterns

- **Positive train Sharpe doesn't mean edge.** Every strategy that posted train
  Sharpe > 1.0 collapsed under walk-forward. Treat full-period positive
  Sharpes as overfitting until proven otherwise.
- **Trade count > 5000 = dead.** Cost drag at our basis kills high-frequency
  strategies. Aim for 50-500 trades over 3 years.
- **49% win rate ≠ edge.** `bb_band_touch_revert` had near-coinflip WR but
  losses 2-3x bigger than wins. Always check profit factor.
- **Sample size matters more than Sharpe.** Train Sharpe 0.47 on 3 trades is
  noise. Train Sharpe 0.30 on 200 trades is a signal worth WF-ing.

---

## 3. Things that worked (infrastructure-wise, not strategy-wise)

- **Walk-forward optimizer** correctly exposed overfitting in `bb_squeeze` and
  `noise_area_breakout`. The reason these strategies looked "almost positive"
  in earlier runs was full-period grid-search overfitting.
- **Market regime classifier** (`bull`/`bear`/`high_vol`/`neutral` from SPY
  50-day trend + 20-day vol) consistently reduces Sharpe drag when used as an
  entry gate. `trend_ride` ungated -3.59 → gated -1.58.
- **Embedding regime gate** (ChromaDB on 60-bar windows) works but didn't
  surface a magic threshold. Best WF on SPY bb_squeeze gave test Sharpe -0.91
  at threshold 0.55.
- **Cost-basis sensitivity** is huge. Dropping COMMISSION from 0.05% to 0.01%
  improved every strategy by ~1 Sharpe.
- **Per-trade CSV dump** (`results/trades/*.csv`) is what makes results
  auditable. Every backtest now writes one when `dump_trades: True`.

---

## 4. The agent loop

End-to-end automation works as of this session. One command:

```bash
python runners/full_auto_pipeline.py --quick --implement-autonomous
```

Runs all 7 agents:

1. **research_agent** — discovery (web) + invention; registry-aware, won't propose duplicates
2. **autonomous_agent** — first-principles invention (no web); also registry-aware
3. **ml_research_agent.research()** — proposes ML approaches; baseline AUC ≈ 0.51 is in the prompt
4. **code_agent** — implements unmatched autonomous ideas; instructed to refuse if Claude detects a duplicate
5. **backtesting_agent** — runs everything with `dump_trades=True`
6. **walk_forward_optimize** — overfit detector on candidates with Sharpe > -1
7. **risk_agent** — gates by `config.RISK` thresholds

Each agent caches its output (`results/research_ideas_round2.json`,
`results/autonomous_ideas.json`, `results/ml_research_approaches.json`).
Re-running without `--refresh-research` skips the SDK calls.

---

## 5. Unexplored leads worth trying

These are NOT confirmed dead. They're untested or barely tested.

- **ML approaches from `ml_research_agent`** (cached in `results/ml_research_approaches.json`):
  - `microstructure_lgbm_meta` (gradient boosting + meta-labeling)
  - `cross_sectional_residual_transformer` (transformer on QQQ-vs-SPY residuals)
  - `temporal_cnn_raw_bars` (dilated CNN on raw bars)
  - `regime_conditional_stacker` (stacked GBM with regime as gate)
  - `options_flow_conditioned_gbm` (would need new data feed)
- **`spy_short_iron_condor_vrp`** from `research_ideas_round2.json` — 0DTE
  iron condor, needs options data + code_agent to implement.
- **The 2 autonomous ideas that didn't bleed catastrophically**:
  - `closing_auction_drift` — DD only -1.2%, could walk-forward with tighter
    params
  - `pdh_pdl_liquidity_grab_fade` — DD only -1.2%, same as above
- **Strategy → ticker matching**. `bb_squeeze` looked positive on MSFT and CAT
  (low-vol large-caps) but failed WF. Could try `noise_area_breakout` on a
  similar low-vol subset.
- **Longer timeframes**. Everything we've tested is 1m. 5m or 15m might
  reduce noise and cost drag together.

---

## 6. Don't repeat these experiments

| Already done | What you'd find |
|---|---|
| Re-run `bb_squeeze` on full universe | Same -0.35 aggregate Sharpe |
| Walk-forward `bb_squeeze` on MSFT/CAT/AAPL/anywhere | Train collapses to negative test |
| Walk-forward `noise_area_breakout` on AAPL | train 1.47 → test -1.68 |
| Backtest at higher commission (0.05%) | Everything ~1 Sharpe worse than now |
| Run `bb_band_touch_revert` at default params | 49% WR, Sharpe ~-12 |
| Run `qqq_spy_dispersion` with ATR stop | Same -22 Sharpe; signal logic is the problem |
| `half_hour_continuation` at default 3bps | Sharpe ~-7 from cost drag |

---

## 7. Project state at this checkpoint

- 14 strategies in `STRATEGIES` registry (10 original + 1 v2 variant + 3
  code_agent-generated)
- 60+ per-trade CSVs in `results/trades/`
- Full auto-pipeline working end-to-end
- 0 strategies passing risk thresholds (Sharpe ≥ 0.8)
- Cost basis: 0.01% commission + 0.02% slippage = 6bps round-trip
- Universe: 20 tickers across tech, financials, industrials, ETFs
- Data: 1-minute bars 2016-2025 from Alpaca parquet dumps

**The honest score:** the *system* works. The *strategies* don't have edge yet.
The next-best move is either (a) implement the ML approaches from
`ml_research_agent` or (b) try a different timeframe / asset class entirely.
Don't keep grinding param sweeps on the existing rule-based strategies — WF
has decisively ruled them out.

---

## 8. BREAKTHROUGH — daily / multi-day holds survive cost (2026-05-28)

The entire history above is **1-minute intraday**, where 6bps round-trip cost
dominates the signal. We finally tested the unexplored lead from section 5
("longer timeframes") and it changes everything.

**Resample the same parquet to DAILY bars and hold for days/weeks** → the 6bps
is amortized over a multi-week move and becomes negligible. Three textbook
daily strategies, long/flat, 6bps costs, 2016–2025, diversified universe
(SPY/QQQ/GLD/MSFT/JPM/GOOGL):

| Daily book | Sharpe | $PnL/$100k | Max DD | WR | Trades | Risk |
|---|---|---|---|---|---|---|
| **blended (all 3)** | **1.07** | +116,503 | −13.9% | 58.8% | 845 | ✅ PASS |
| rsi2_meanrev (Connors RSI-2) | 0.87 | +61,776 | −9.7% | 68.1% | 407 | ✅ PASS |
| trend_5020 (50/200 SMA) | 1.07 | +287,217 | −28.3% | 76.5% | 34 | ✗ DD+trades |
| donchian (20/10 breakout) | 0.51 | +51,359 | −28.7% | 48.0% | 404 | ✗ Sharpe+DD |

- **In-sample 0.84 → out-of-sample 1.72** (the OPPOSITE of every 1m strategy).
- Blended book positive in 4 of 5 walk-forward folds; the one negative fold is
  the 2022 bear (−3.1% vs market ~−25%). RSI-2 positive in **all 5 folds**.
- Robust across universes: on all 20 tickers the blended book is Sharpe **1.42**
  (DD −18% there, only fails the gate because of NVDA/TSLA/CVNA single-name DD).

**Code:** `agents/daily_strategies.py` (library + $100k portfolio backtester,
authoritative numbers), `runners/daily_book.py` (board report + RiskAgent),
`runners/daily_rebalance.py` (Alpaca paper deploy, pulls fresh yfinance data).
Registered in STRATEGIES as `daily_rsi2_meanrev` / `daily_donchian` /
`daily_trend_5020` (ATR stop + max-hold disabled — daily strategies self-exit).

**Caveats:** long-only (equity beta), bull-market sample, single OOS window.
The edge is real but it's a long-biased trend/reversion book — value is
drawdown reduction + regime resilience, not market-neutral alpha.

**Don't:** force these through the 1m intraday engine with ATR stops enabled —
the intraday stop fires constantly on resampled daily signals and wrecks them.

### 8a. RSI-2 universe experiments (2026-05-29) — breadth does NOT beat the curated book

Tuned RSI-2 (entry RSI(2)<30, exit>50, 100d trend filter) via anchored
walk-forward clears >=100 trades/yr: OOS Sharpe 1.09, +$48.6k, -9.7% DD, 112
trades/yr on the 6-name book. Settings in `WALK_FORWARD_SETTINGS.md`.

Tried scaling RSI-2 to the whole S&P 500 (free yfinance daily; `data/sp500.py`,
`runners/sp500_rsi2.py`):
- Full 500 diversified (1/N): Sharpe 0.585 / OOS 1.01, -8% DD, +$29k — fails 0.8 gate (mostly cash, tiny weights).
- Concentrated (top-K oversold): more return (+$112-156k) but -38% to -48% DD — fails badly.
- **Low-vol filter (40 calmest + ETF anchors): Sharpe 0.336 — WORSE.** Low-vol names have small bounces that don't clear 6bps cost.
- **Verdict: the curated 6-name book (SPY/QQQ/GLD/MSFT/JPM/GOOGL at 1/6 weight) wins (Sharpe 0.95-1.09).** Single-stock RSI-2 is noisier than index-ETF RSI-2; breadth dilutes risk-adjusted quality. Don't re-run universe-expansion expecting a Sharpe gain.

### 8b. The 10-20%/yr gap (open)

Gate-passing books return 6-9%/yr (blended 8.6%, RSI-2 5.8%). 50/200 trend hits
15.1% but at -28% DD (fails -15% gate). Reaching 10-20%/yr at acceptable DD is
unsolved — needs either (a) trend-tilted blend, (b) modest leverage on the
blended book (1.3-1.5x → ~11-13% at -18 to -21% DD), or (c) a new daily
mechanism. This is the current open problem.

### 8d. DATA SOURCE FIX — daily book now uses ADJUSTED yfinance (2026-05-29)

The local 1m parquet is RAW/unadjusted. `runners/verify_trades_vs_yfinance.py`
caught it: only 21.8% of daily-book fills landed in yfinance's adjusted
high-low range; GOOGL's unadjusted 20:1 split (2022) corrupted its series.
**Fix:** `daily_bars()` now sources split/dividend-ADJUSTED yfinance daily bars
(env `DAILY_USE_ADJUSTED=0` forces raw parquet). After the fix, verification =
100% in-range / 0.00% diff, and the blended book Sharpe ROSE 1.09 -> 1.32 (the
bad data had been a drag). All parquet names with splits (GOOGL/AAPL/AMZN/TSLA/
NVDA) are now safe. Refreshed board numbers: blended Sharpe 1.32, +$178.7k,
10.4% CAGR, -13.4% DD (2016-2026, PASSES risk — finally hits the 10-20% target).
Full S&P 500 (503 names, adjusted, 1/N): blended Sharpe 0.93, -14.1% DD, 5.8%
CAGR — passes but lower return (breadth dilutes at 1/N). Curated 6-name wins.

### 8e. New-strategy research + blend tuning (2026-05-29)

Tested 3 new daily candidates on the quality-10 (`runners/strategy_lab.py`,
`CANDIDATE_STRATEGIES` in daily_strategies). A new sleeve only helps if it's
UNCORRELATED to the core — measured the return-correlation matrix:
- **abs_momentum** (6mo time-series momentum): standalone Sharpe 1.20 BUT **0.93
  correlation with trend_5020** — it's not a new edge, just more trend exposure.
  Adding it raises return (10.6% CAGR) and DD (-14.5%) — equivalent to leverage.
- **turn_of_month** (month-end seasonality): the BEST diversifier (corr ~0.27 to
  core). Cuts the blend's DD from -11.8% -> -8.3% at ~same Sharpe. Promoted as
  the `defensive` book in daily_rebalance.
- **zscore_revert**: weak (Sharpe 0.52) and 0.72-correlated to trend — skipped.
- **inverse-vol (risk-parity) weighting** of the core: DD -11.8% -> -9.1%, Sharpe
  ~unchanged (1.21). A clean low-DD lever.
- **No combination materially beats the current blend's Sharpe (~1.23)** — the
  3-core blend is near the efficient frontier for these mechanisms.

### 8f. CROSS-SECTIONAL book + dual momentum (2026-05-29) — improves the blend

Built a portfolio-level cross-sectional book (`backtest_cross_sectional`, rank
universe daily, long top-k, equal weight). On quality-10:
- raw cross-sectional momentum (12-1, k=3): HIGHEST return of anything — Sharpe
  1.01, **+$672k, 21.8% CAGR** — but -29.5% DD (fails gate); corr 0.82 to blend.
- **dual momentum** (add a market filter: cash when SPY < 200-SMA): Sharpe 1.08,
  +$449k, 17.8% CAGR, DD cut to -23.4%, corr to blend drops to 0.70.
- cross-sectional REVERSAL: weak (Sharpe ~0.7, -35% DD) — universe too small.
- **Adding dual-momentum to the core-3 blend IMPROVES it:** Sharpe 1.23->1.26,
  CAGR 9.3%->11.5%, DD -11.8%->-13.9% (still PASSES). This is the new best
  deployable — `--book blended_plus` (wired into daily_rebalance with a live
  cross-sectional sleeve). Finally hits the 10-20% target inside the -15% gate.

### 8g. Vol-targeting rescues the high-return strategies (2026-05-29)

The high-PnL books (trend_5020, cross-sectional momentum/dual-mom) failed the
gate only on DRAWDOWN. A VOLATILITY-TARGETING overlay (`vol_target()` in
daily_strategies: scale exposure to a target annualized vol using yesterday's
realized vol, cap at max_leverage) fixes it AND raises Sharpe:
- trend_5020: 1.12 / 15.3% / -24.9%  ->  vt12%: **1.36 / 13.7% / -12.7%** PASS
- xs_momentum 12-1: 1.01 / 21.8% / -29.5%  ->  vt10%: **1.25 / 13.4% / -14.8%** PASS
- xs_dualmom 12-1: 1.08 / 17.8% / -23.4%  ->  vt10%: 1.22 / 12.3% / -15.2% (marginal)
All vol-targeted versions are positive in 5/5 walk-forward folds (IS->OOS holds
or improves). Tooling: `runners/derisk_wf.py`.
- FULL S&P 500 cross-sectional + vt10%: Sharpe 1.19, 12.6% CAGR, -13.0% DD,
  5/5 folds (raw was +$6M / 48.8% CAGR / -43.5% DD — un-risk-managed artifact).
- Deployable: `daily_rebalance.py --xs-universe sp500` (rank top-10 momentum
  across the index) and `--vol-target 0.12 --max-leverage 1.0` (de-risk overlay).
  Works on any --book. blended_plus + full-500 xs sleeve verified live.

### 8c. Cleanup (2026-05-29)

Removed 47 unprofitable generated modules from `strategies/`. Their names +
verdicts are retained in `results/strategy_ledger.json` (61 entries) so agents
know what's been tried; the ledger is injected into agent prompts via
`_existing_strategies_summary()`. The 3 daily winners live in
`agents/daily_strategies.py`.
