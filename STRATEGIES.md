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
| **crypto_trend** | abs-momentum on BTC+ETH | `lookback=126` | **PASSES @≤5%**: Sharpe →1.63, CAGR →21.6%, DD −14.2% — but 45% standalone CAGR is a one-time secular bull; governance-gated |

*Finding 1 — equity-pattern saturation: every new long/flat price-pattern sleeve on the
quality-10 overlaps the six deployed ones and adds only noise.*
*Finding 2 — asset-class diversifiers (bonds/commodities/intl/all-weather) all REDUCE
return for marginal drawdown help; the vol-target overlay already provides cheaper crash
protection. The only additive sources found: the **low-vol equity factor** (modest, safe)
and a small **crypto-trend** sleeve (powerful but backward-looking + governance-gated).*
*Finding 3 — ML trade-failure study (`runners/trade_failure_ml.py`): walk-forward logistic
model on entry features gives OOS AUC ≈ 0.50 (RSI-2) / 0.53 (Donchian) — i.e. trade
outcomes are **not predictable** from entry conditions. No ML entry-filter is wired; the
deployed sleeves are already efficient (a healthy negative result, not a curve-fit).*

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

## Options income sleeves (no leverage) — `runners/options_income.py`
Harvest the **volatility risk premium** (option buyers overpay for insurance: implied
vol > realized vol). A *different return source* from the price-pattern sleeves —
which is why they add value where new equity sleeves don't. Both fully collateralized
(no leverage, no naked short risk). 1-month cycles on SPY+QQQ.

| Sleeve | Mechanism | CAGR / vol / Sharpe / DD | Notes |
|---|---|---|---|
| **putwrite** | sell 2% OTM cash-secured put monthly | 9.2% / 7.2% / 1.26 / −11.1% | defensive income; ~half SPY's vol & DD; corr 0.76 |
| **buywrite** | own SPY/QQQ, sell 2% OTM call monthly | 16.9% / 11.5% / 1.43 / −16.8% | smoother equity; corr 0.90 |

> **MODELED, not a fill backtest.** No historical option chains available → premiums
> are Black-Scholes with IV = realized vol + a **volatility-risk-premium markup** (`--vrp`,
> default 3 vol points, empirically ~what SPX IV−RV has averaged). Sensitivity at 0pt
> markup: PutWrite Sharpe 0.80, BuyWrite 1.11 — i.e. the edge scales with that one
> assumption, shown in full in the runner's output. **Validate live on Alpaca paper**
> (`agents/options_agent.py`) before any real capital. Not yet wired to the live book.
>
> **Best fit:** the put-write is an upgrade path for the *idle-cash → BIL* overlay
> (earn ~9% on defensive collateral vs ~4% T-bills), still no leverage.

Run: `python runners\options_income.py --kind putwrite --tickers SPY QQQ`
(or `--kind buywrite`; `--vrp 0.02..0.04` to stress the premium assumption).

## Honest caveats
Long-biased equity book, validated 2016–2026 (one decade, one out-of-sample window).
Not market-neutral; lean years (2018-style) are cushioned by cash yield + recovery
sleeve but still low-single-digit on the strategy side. Live PEAD caps to 25 names
(vs uncapped backtest). Paper-trade before real capital. See BOARD_SUMMARY.md and
LESSONS.md for the full record.
