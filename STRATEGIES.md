# Strategy Reference — every strategy, sleeve, overlay & parameter

All strategies trade **daily / multi-day holds** on **split/dividend-adjusted** data,
**6 bps round-trip** cost, $100k base. Risk gate: Sharpe ≥ 0.8, max DD ≥ −15%,
win-rate ≥ 45%, trades ≥ 50. Code: `agents/daily_strategies.py`; execution:
`runners/daily_rebalance.py`; new-strategy screening: `runners/portfolio_allocator.py`.

---

## Deployed book: `portfolio_full`

7 sleeves + 3 overlays. **Sharpe 1.53 · CAGR 18.4% · max DD −13.4% · 2018–2020 +9% · positive in 5/5 walk-forward folds.**

```
sleeve weights:  rsi2_meanrev 0.252 · donchian 0.198 · trend_5020 0.126
                 xs_dualmom 0.072 · recovery 0.162 · pead 0.090
                 lowvol_def 0.10   (six price sleeves scaled to 90%, lowvol 10%)
overlays:        vol-target 17% (≤1.8× leverage) · idle cash → BIL T-bills
                 · early-warning de-risk
deploy:  python runners\daily_rebalance.py --book portfolio_full \
              --xs-universe sp500 --vol-target 0.17 --max-leverage 1.8 --live
```

Leverage note: the 1.8× cap (raised from 1.6×) deploys the risk budget the
early-warning overlay freed up — it only levers that high when realized vol is
low, and de-levers (vol-target) + cuts to 60% (early-warning) when vol spikes.
The −13.1% backtest DD already includes the COVID-2020 crash and the 2022 bear.
Residual risk: an unprecedented one-day gap hurts ~1.8× as much.

---

## Sleeves (signal strategies)

| Sleeve | Mechanism | Universe | Key parameters | Standalone (Sharpe/CAGR/DD) |
|---|---|---|---|---|
| **rsi2_meanrev** | buy short-term dips in an uptrend (Connors RSI-2) | quality-10 | `rsi_period=2, entry_rsi=30, exit_rsi=50, trend_sma=100` | 0.87 / 5.1% / −9.0% |
| **donchian** | 20-day-high breakout, exit 10-day low | quality-10 | `entry_lookback=20, exit_lookback=10` | 0.94 / 7.1% / −12.8% |
| **trend_5020** | 50/200-day SMA trend filter | quality-10 | `fast=50, slow=200` | 1.12 / 15.3% / −24.9% |
| **xs_dualmom** | cross-sectional 12-1 momentum, top-K, cash in bear | **full S&P 500** | `lookback=252, skip=21, k=10, market_filter SPY>200d` | 1.26 / 36.2% / −34.4% |
| **recovery** | catch bull-run snapbacks: reclaim 50d after below 200d, hold | quality-10 | `hold_days=120` | 0.90 / 9.8% / −21.5% |
| **pead** | post-earnings drift: buy gap-up beats, hold the drift | **full S&P 500** | `gap_pct=0.05, vol_mult=2.0, hold_days=60` (live: 25 freshest) | 1.10 / 5.0% / −10.2% |
| **lowvol_def** | hold 30 lowest-realized-vol names while SPY>200d, else → BIL | **full S&P 500** | `vol_window=60, k=30, market_filter SPY>200d` | 1.09 / 10.0% / −12.8% |

### Candidate sleeves (in the allocator pool; deploy only if they pass the gate)
| Sleeve | Mechanism | Parameters | Status |
|---|---|---|---|
| trend_multi | multi-speed trend (avg of fast/med/slow crosses) | `speeds=[(20,100),(50,200),(100,300)]` | better trend sleeve; washes out in portfolio |
| turn_of_month | long the turn-of-month window | `pre=1, post=3` | low edge; in `defensive` book |
| zscore_revert | z-score mean reversion vs N-day mean | `lookback=20, entry_z=-2, exit_z=0, trend_sma=200` | rejected (weak) |
| abs_momentum | time-series momentum, long if >0 | `lookback=126` | redundant with trend |
| capitulation | buy extreme oversold (no trend filter) | `entry_rsi=5, exit_rsi=55, drop_pct=0.07` | rejected (no edge) |
| cross-sectional reversal | buy biggest losers | `lookback=3-5, k=30` | rejected (−35% DD) |
| managed-futures (proxy) | long/short TS-momentum across asset ETFs | 12-mo sign, inverse-vol | rejected (dilutive); see MANAGED_FUTURES_PROPOSAL.md |
| high_momentum | 52-week-high proximity momentum (George-Hwang) | `lookback=252, near_pct=0.05, exit_pct=0.15` | rejected: Sharpe 1.05 but DD −15.0% + slightly *hurts* book (overlaps trend/xs) |
| bollinger_revert | buy lower Bollinger band in uptrend, exit mid-band | `window=20, num_std=2, trend_sma=200` | rejected: Sharpe 0.60 (overlaps RSI-2) |
| ma_pullback | buy pullback to 20-day MA in a 50/200 uptrend | `pull_sma=20, fast=50, slow=200, target_pct=0.03` | rejected: DD −20.8% (overlaps RSI-2) |
| bond_trend | abs-momentum on TLT/IEF | `lookback=126` | rejected: −0.19 corr but costs ~2pt CAGR for 1pt DD (bad trade) |
| commodity_trend | abs-momentum on GLD/DBC/SLV | `lookback=126` | rejected: +0.02 Sharpe (noise) |
| intl_trend | abs-momentum on EFA/EEM/VEA/VWO | `lookback=126` | rejected: corr 0.56, −0.03 Sharpe |
| allweather_trend | abs-momentum on 7 asset-class ETFs | `lookback=126` | rejected: costs ~1pt CAGR for 0.5pt DD |
| **lowvol_factor** | hold 30 lowest-realized-vol S&P names, monthly | `vol_window=60, k=30` | **PASSES** @12%: Sharpe 1.46→1.53, CAGR 18.6%, DD −14.1% — clean, board-friendly |
| **crypto_trend** | abs-momentum on BTC+ETH | `lookback=126` | **PASSES @≤5%**: Sharpe →1.69, CAGR →21.8%, DD −13.7% — WIRED as opt-in (`--crypto-sleeve`, default OFF); 45% standalone is a one-time secular bull; **governance-gated** |
| recovery_xs | recovery on full S&P 500 (cross-sectional) | `hold_days=120, cap=0.10` | rejected: CAGR 15% but DD −39%, lowers book Sharpe; corr 0.72 to recovery (redundant) |
| xs concentration | momentum top-5 vs top-10 | `k=5` | rejected: top-10 already Sharpe-optimal; top-5 = −45% DD for no Sharpe gain |
| rsi2_xs | RSI-2 mean-reversion on full S&P 500 | `entry_rsi=30, cap=0.05` | rejected: Sharpe 0.50, DD −42% (catches falling knives; the quality-10 filter is what makes RSI-2 work) |

*Finding 1 — equity-pattern saturation: every new long/flat price-pattern sleeve on the
quality-10 overlaps the six deployed ones and adds only noise.*
*Finding 2 — asset-class diversifiers (bonds/commodities/intl/all-weather) all REDUCE
return for marginal drawdown help; the vol-target overlay already provides cheaper crash
protection. The only additive sources found: the **low-vol equity factor** (modest, safe)
and a small **crypto-trend** sleeve (powerful but backward-looking + governance-gated).*
*Finding 3 — Machine learning adds nothing here, confirmed two ways:*
*(a) Trade-failure models (`runners/trade_failure_ml.py`): both logistic AND gradient
boosting give OOS AUC ≈ 0.50 (RSI-2) / 0.50–0.53 (Donchian) — trade outcomes are **not
predictable** from entry conditions.*
*(b) ML alpha model (`runners/ml_alpha.py`): a walk-forward HistGradientBoosting return-
predictor (7 factors, top-30, market-filtered) gets OOS Sharpe 0.98 / −33% DD — **worse**
than the rule-based momentum sleeve (Sharpe 1.44 / −18%). Tabular financial data is low
signal-to-noise; the hand-built signals already capture the edge and ML overfits. No ML
model deployed — a credible negative, not a curve-fit.*

---

## Books (sleeve combinations)

| Book | Weights | Sharpe / CAGR / DD | Use |
|---|---|---|---|
| **`portfolio_full`** ⭐ | 6 price sleeves ×0.90 + lowvol_def .10 | 1.53 / 18.4% / −13.4% | **deployed** @ vt 17% / 1.8× — best all-round |
| `portfolio_rec` | rsi .32, don .24, trd .16, xs .08, rec .20 | 1.43 / 17.1% / −14.1% | max lean-year capture |
| `portfolio_div` | rsi .35, don .27, trd .15, xs .08, pead .15 | 1.47 / 16.0% / −12.3% | smoothing via PEAD |
| `portfolio` | risk-parity rsi .41, don .32, trd .18, xs .09 | 1.39 / 16.2% / −13.0% | core risk-parity |
| `blended_plus` | rsi/don/trd/xs 0.25 each | 1.44 / 14.5% / −12.7% | no-leverage option |
| `blended` | rsi/don/trd 1/3 each | 1.23 / 9.3% / −11.8% | conservative core-3 |
| `defensive` | rsi/don/trd/turn_of_month 0.25 each | 1.22 / 8.3% / −8.3% | lowest drawdown |
| `trend_tilt` | trend 0.5, rsi 0.5 | 1.15 / ~12% / −17% | trend-heavy (fails gate) |
| `regime_adaptive` | weights+leverage shift by SPY regime | 1.4× / up to 20% / −18% | aggressive (leverage opt-in) |
| `pead` | 100% PEAD (25 freshest) | 1.10 / 5.0% / −10.2% | standalone event sleeve |

---

## Overlays (applied on top of any book)

| Overlay | Rule | Effect |
|---|---|---|
| **Vol-targeting** | scale exposure so realized vol ≈ target (`--vol-target 0.15`, `--max-leverage 1.6`); de-levers when vol rises | controls drawdown; conditional leverage in calm markets |
| **Idle-cash → T-bills** | park `1 − invested` in **BIL** (`--park-cash BIL`) | riskless yield (~4–5%) on idle capital, esp. in lean years |
| **Early-warning de-risk** | cut exposure to 60% when **SPY < 50-day AND 20-day vol > 20%** | front-runs the lagging 200-day bear signal (Sharpe 1.45→1.48, DD −13.8%→−11.7%) |
| **Dual-momentum filter** | xs sleeve holds only when **SPY > 200-day** | cross-sectional sleeve goes to cash in bears |
| **Regime detection** | SPY vs 200-day + 20-day vol → BULL_CALM / BULL_VOL / BEAR | printed each run; drives `regime_adaptive` |
| **No-trade band** | skip reconcile orders < $250 | controls churn/cost |
| **Fractional orders** | dollar-sized (notional) market orders | exact weights on high-priced names |

---

## Regime handling (bull ↔ bear)
- **Bull → bear** (SPY crosses below 200-day): trend/momentum sleeves → cash, recovery dormant, RSI-2 blocked below trend, vol-target de-risks, early-warning cuts to 60%, cash → T-bills. (Why the book made +33% in the 2022 bear.)
- **Bear → bull**: the **recovery** sleeve fires when price reclaims the 50-day after being below the 200-day — catching snapbacks (early-2019, spring-2020).
- The 200-day is *lagging* (confirms a bear after ~10–15% drop); the early-warning + vol-target reduce that lag.

## Universe & data
- **Per-ticker sleeves** (rsi2, donchian, trend, recovery): quality-10 = SPY, QQQ, GLD, MSFT, AAPL, GOOGL, AMZN, JPM, UNH, XOM.
- **Cross-sectional sleeves** (xs_dualmom, pead): full S&P 500 (`--xs-universe sp500`).
- Data: split/dividend-adjusted daily bars (yfinance); `DAILY_USE_ADJUSTED=0` forces raw parquet.

## Options income sleeves (no leverage) — REJECTED on proper analysis
Harvest the **volatility risk premium** by selling options. Two versions built:
`runners/options_income.py` (v1, fixed-% strikes) and `runners/options_income_v2.py`
(v2, **done properly**: delta-targeted strikes + market filter so you never write
into a downtrend; honest non-overlapping windows).

**Verdict: does NOT clear the bar.** The proper delta-strike sweep (cash-secured
put-write, SPY+QQQ, VRP 3pt):

| Put delta | CAGR | Sharpe | Max DD |
|---|---|---|---|
| 0.10 | 1.7% | 0.27 | −24.0% |
| 0.25 | 5.1% | 0.60 | −25.9% |
| 0.45 (near money) | 9.7% | 0.96 | −27.0% |

Every strike caps at **Sharpe < 1.0 with −24% to −27% drawdown** — it would break
the −15% gate, Sharpe is far below the book's 1.53, and it's 0.76-correlated to SPY
(not a diversifier). The VRP is real but it is **payment for selling crash insurance**:
you get assigned exactly when the market gaps down, *adding* equity-crash exposure we
already have. **v1's rosy 9.2%/Sharpe 1.26/−11% was a WINDOWING ARTIFACT** (its 21-day
blocks straddled the COVID crash and skipped the assignment); the honest v2 surfaces
the true −25% tail. Not deployed. (Still MODELED — no historical chains — but the
conclusion holds regardless of the VRP assumption since DD is the binding problem.)

Run: `python runners\options_income_v2.py` (`--delta 0.10..0.45`, `--vrp` to stress).

## Regime coverage & why there's no crash-hedge (`runners/regime_coverage.py`)
Audit of the deployed book by regime (trend × vol), 2016–2026:

| Regime | % days | SPY ann | Book ann | Coverage |
|---|---|---|---|---|
| Bull · calm | 70% | +20.5% | +25.2% | ✅ beats market |
| Bull · stormy | 7% | +80.5% | +35.9% | ⚠️ lags violent rallies (vol-target caps), still +36% |
| Bear · calm | 10% | −2.9% | −9.8%* | ✅ not a gap — see note |
| Bear · stormy / crash | 13% | −28.0% | −12.5% | ✅ cushioned (−1.7% on worst-1% days) |

*The Bear·calm −9.8% looked like a gap but a sub-bucket diagnosis
(`grind_fix_test.py` + diagnosis) shows it is **not**: in the genuine grind-down
(below both 50d & 200d) the book loses −36% ann vs SPY −57% (it loses *less*); the
−9.8% is dominated by the book being prudently cautious in **below-200d recovery
bounces** (captured +8% of a +34% rally — a lag, not a loss) plus a small-sample
annualization artifact (2016 was +2.9%). Three de-risk fixes (200d leverage cap,
dual-MA grind de-risk, protective puts) all REDUCED risk-adjusted return without
improving it — there is no downside hole to plug here.

Up-capture 68% / down-capture 60%; book max DD −13.4% vs SPY −33.7%.
**The book cushions crashes, it does not profit from them** (it's long-biased).
Two ways to add crash protection were tested and BOTH rejected:
- *Trend-aware leverage cap* (no leverage below 200d, `runners/regime_fix_test.py`):
  backfired — cut CAGR 18.4%→17.7% and made Bear·calm worse, because the leverage
  it removes also funds the below-200d recovery bounces.
- *Protective-put overlay* (`runners/tail_hedge.py`): costs up to 2.6pt CAGR and
  only helps a fast single-month crash; slow bears (2018/2022) get worse from
  premium bleed. Not worth it — the existing overlays cover the downside cheaper.

Conclusion: the book is at its efficient frontier for daily, mostly-no-leverage
strategies on this universe (~26 ideas tested). The deployed sleeve WEIGHTS were
also walk-forward validated (`runners/weight_optimize.py`): a Sharpe-maximizing
optimizer cannot beat the hand-set weights out-of-sample (avg OOS Sharpe 1.37 opt
vs 1.38 current) and overfits badly (drops RSI-2 to 0.3%). Next gains come from a
live paper track record + monitoring, not strategy #27.

## Fundamental quality screen (Finnhub) — live tilt, NOT backtested
New data source: `data/finnhub_fundamentals.py` + `runners/fundamental_screen.py`.
Pulls current company financials (ROE, net margin, current ratio, debt/equity,
revenue growth, P/E) and builds a cross-sectional **quality/value composite**
(mean of signed z-scores). Intended use: a **quality filter on the momentum sleeve**
("quality momentum" — keep high-momentum names that are *also* high-quality; drop
the expensive, fragile ones like a P/E-346 momentum chaser).

> **HONEST DATA LIMIT:** Finnhub's free tier returns the *current* fundamental
> snapshot, **not point-in-time history**. So this is a **live screen/tilt only** —
> a rigorous backtested fundamental factor needs paid point-in-time data (else
> look-ahead bias). It is **not deployed** to the live book; it's a tool to
> rank/filter today's holdings. Wiring it as a live momentum filter is an option
> pending either paid history or a forward live-validation period.

Run: `python runners\fundamental_screen.py` (default: quality-10 + live momentum picks).

**Wiring tested & SKIPPED:** the fundamental filter can't be backtested (no free
point-in-time data), so the concept was tested with a backtestable quality proxy
(low realized vol) on the momentum sleeve (`runners/quality_momentum_test.py`):
quality-filtering momentum CUT return 47.6%→15.3% and Sharpe 1.33→0.97 — the biggest
momentum winners *are* the volatile names, so the screen removes the engine. Not
wired to the live book. The fundamental screen remains a live inspection tool only.

## Honest caveats
Long-biased equity book, validated 2016–2026 (one decade, one out-of-sample window).
Not market-neutral; lean years (2018-style) are cushioned by cash yield + recovery
sleeve but still low-single-digit on the strategy side. Live PEAD caps to 25 names
(vs uncapped backtest). Paper-trade before real capital. See BOARD_SUMMARY.md and
LESSONS.md for the full record.
