# Walk-Forward Settings — 3 Daily Algorithms

**Generated:** 2026-05-29  
**Universe:** SPY, QQQ, GLD, MSFT, JPM, GOOGL  
**Method:** anchored expanding-window walk-forward (train 2016→Y-1, test Y, roll forward)  
**Manager constraint:** >= 100 trades/year, enforced during in-sample parameter selection  
**Costs:** 6 bps round-trip · **Start capital:** $100,000

> Out-of-sample = stitched test years only, each traded with parameters chosen *before* that year. This is the honest number.


## rsi2_meanrev

**Recommended deploy parameters** (trained on all data through the last full year):

```json
{
  "rsi_period": 2,
  "entry_rsi": 30,
  "exit_rsi": 50,
  "trend_sma": 100
}
```
- meets >= 100 trades/yr: **YES**

**Out-of-sample (walk-forward) performance:**

| Sharpe | $PnL | Total ret | Max DD | Win rate | Trades/yr |
|--------|------|-----------|--------|----------|-----------|
| 1.087 | $48,570 | 48.57% | -9.7% | 64.5% | 111.5 |

**Per-fold detail:**

| Test yr | Train | Params | Train SR | Train tr/yr | OOS SR | OOS ret | OOS tr/yr | floor |
|---------|-------|--------|----------|-------------|--------|---------|-----------|-------|
| 2020 | 2016-01-04..2019-12-31 (4.0y) | rsi_period=2, entry_rsi=30, exit_rsi=50, trend_sma=100 | 0.684 | 109.0 | 0.743 | 7.12% | 124.1 | Y |
| 2021 | 2016-01-04..2020-12-31 (5.0y) | rsi_period=2, entry_rsi=30, exit_rsi=50, trend_sma=200 | 0.684 | 113.2 | 1.545 | 11.61% | 145.5 | Y |
| 2022 | 2016-01-04..2021-12-31 (6.0y) | rsi_period=2, entry_rsi=30, exit_rsi=50, trend_sma=200 | 0.827 | 118.5 | -0.793 | -3.21% | 32.1 | Y |
| 2023 | 2016-01-04..2022-12-31 (7.0y) | rsi_period=2, entry_rsi=30, exit_rsi=50, trend_sma=100 | 0.689 | 104.3 | 1.584 | 9.09% | 123.4 | Y |
| 2024 | 2016-01-04..2023-12-31 (8.0y) | rsi_period=2, entry_rsi=30, exit_rsi=50, trend_sma=100 | 0.789 | 106.6 | 1.875 | 11.18% | 143.1 | Y |
| 2025 | 2016-01-04..2024-12-31 (9.0y) | rsi_period=2, entry_rsi=30, exit_rsi=50, trend_sma=100 | 0.901 | 110.7 | 1.664 | 5.86% | 95.0 | Y |

## donchian

**Recommended deploy parameters** (trained on all data through the last full year):

```json
{
  "entry_lookback": 55,
  "exit_lookback": 20
}
```
- meets >= 100 trades/yr: **NO (floor not reachable)**

**Out-of-sample (walk-forward) performance:**

| Sharpe | $PnL | Total ret | Max DD | Win rate | Trades/yr |
|--------|------|-----------|--------|----------|-----------|
| 0.215 | $10,388 | 10.39% | -33.7% | 44.7% | 34.8 |

**Per-fold detail:**

| Test yr | Train | Params | Train SR | Train tr/yr | OOS SR | OOS ret | OOS tr/yr | floor |
|---------|-------|--------|----------|-------------|--------|---------|-----------|-------|
| 2020 | 2016-01-04..2019-12-31 (4.0y) | entry_lookback=20, exit_lookback=5 | 0.89 | 64.4 | 0.684 | 6.02% | 67.0 | n |
| 2021 | 2016-01-04..2020-12-31 (5.0y) | entry_lookback=20, exit_lookback=20 | 0.835 | 25.2 | 1.443 | 13.57% | 27.1 | n |
| 2022 | 2016-01-04..2021-12-31 (6.0y) | entry_lookback=20, exit_lookback=20 | 0.933 | 25.5 | -1.755 | -29.65% | 25.1 | n |
| 2023 | 2016-01-04..2022-12-31 (7.0y) | entry_lookback=55, exit_lookback=5 | 0.698 | 44.6 | 0.67 | 3.74% | 50.2 | n |
| 2024 | 2016-01-04..2023-12-31 (8.0y) | entry_lookback=55, exit_lookback=20 | 0.71 | 18.0 | 0.983 | 9.5% | 16.0 | n |
| 2025 | 2016-01-04..2024-12-31 (9.0y) | entry_lookback=55, exit_lookback=20 | 0.745 | 17.8 | 2.767 | 14.71% | 17.8 | n |

## trend_5020

**Recommended deploy parameters** (trained on all data through the last full year):

```json
{
  "fast": 20,
  "slow": 100
}
```
- meets >= 100 trades/yr: **NO (floor not reachable)**

**Out-of-sample (walk-forward) performance:**

| Sharpe | $PnL | Total ret | Max DD | Win rate | Trades/yr |
|--------|------|-----------|--------|----------|-----------|
| 0.938 | $110,773 | 110.77% | -28.3% | 51.0% | 8.7 |

**Per-fold detail:**

| Test yr | Train | Params | Train SR | Train tr/yr | OOS SR | OOS ret | OOS tr/yr | floor |
|---------|-------|--------|----------|-------------|--------|---------|-----------|-------|
| 2020 | 2016-01-04..2019-12-31 (4.0y) | fast=50, slow=200 | 1.324 | 3.5 | 0.64 | 15.09% | 4.0 | n |
| 2021 | 2016-01-04..2020-12-31 (5.0y) | fast=10, slow=150 | 1.101 | 6.0 | 2.002 | 27.85% | 6.0 | n |
| 2022 | 2016-01-04..2021-12-31 (6.0y) | fast=10, slow=150 | 1.258 | 6.0 | -2.399 | -13.04% | 10.0 | n |
| 2023 | 2016-01-04..2022-12-31 (7.0y) | fast=10, slow=100 | 0.979 | 10.7 | 1.685 | 20.6% | 17.1 | n |
| 2024 | 2016-01-04..2023-12-31 (8.0y) | fast=10, slow=100 | 1.073 | 11.5 | 1.536 | 19.24% | 7.0 | n |
| 2025 | 2016-01-04..2024-12-31 (9.0y) | fast=20, slow=100 | 1.139 | 7.3 | 2.122 | 14.55% | 7.4 | n |