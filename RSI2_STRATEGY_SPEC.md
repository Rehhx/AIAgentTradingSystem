# RSI-2 Mean-Reversion — Strategy Spec (for the Go port)

Language-neutral specification of the deployed RSI-2 book. Implementing exactly
what's below in Go will reproduce the Python backtest. Reference implementation:
`agents/daily_strategies.py` (`sig_rsi2_meanrev`, `_rsi`, `sleeve_returns`).

---

## 1. What it is (plain English)

Larry Connors' **RSI(2)** short-term mean reversion: in a confirmed uptrend, buy
sharp 2-day pullbacks and sell into the bounce. "Buy the dip, but only when the
longer trend is up." It is **long/flat only** (never short), holds for a few days
on average, and sits in cash when there's no qualifying dip.

- **Timeframe:** daily bars (one bar per trading day).
- **Trend filter** keeps you out of falling markets (this is what produced only
  a −1.9% loss in the 2022 bear instead of riding it down).
- **Universe:** equal-weight across 6 names — SPY, QQQ, GLD, MSFT, JPM, GOOGL.

---

## 2. Deployed parameters

| Param | Value | Meaning |
|---|---|---|
| `rsi_period` | **2** | RSI lookback (Wilder) |
| `entry_rsi` | **30** | enter long when RSI(2) < 30 |
| `exit_rsi` | **50** | exit to flat when RSI(2) > 50 |
| `trend_sma` | **100** | only buy when close > 100-day SMA |
| cost | **6 bps** | round-trip (3 bps per side) |

These are the walk-forward-optimized settings that clear ≥100 trades/year
(~109/yr) at out-of-sample Sharpe 1.09. The classic Connors defaults are
`entry=10, exit=70, trend_sma=200`; we loosened entry/exit to trade more often.

---

## 3. Indicators

### RSI(2) — Wilder smoothing via recursive EMA (alpha = 1/period)

For `period = 2`, `alpha = 0.5`. Walk forward through the close series:

```
change[t] = close[t] - close[t-1]
gain[t]   = max(change[t], 0)
loss[t]   = max(-change[t], 0)

# seed at the first delta (t = 1), then recurse:
avgGain[t] = alpha*gain[t] + (1-alpha)*avgGain[t-1]
avgLoss[t] = alpha*loss[t] + (1-alpha)*avgLoss[t-1]

RSI[t] = 100                       if avgLoss[t] == 0
       = 100 - 100/(1 + avgGain[t]/avgLoss[t])   otherwise
```

(`adjust=False` EMA. Seeding only perturbs the first handful of bars, which are
inside the 100-day warmup and never traded — so it doesn't affect signals.)

### Trend filter — simple moving average

```
SMA100[t] = mean(close[t-99 .. t])      # needs >=100 bars of history
aboveTrend[t] = close[t] > SMA100[t]
```

---

## 4. Signal / state machine (per ticker)

Long/flat state, evaluated each day. `position` starts at 0 (flat).

```
for t in 0 .. N-1:
    if t < trend_sma:                 # warmup, no position
        position = 0
    else:
        enter = (RSI[t] < entry_rsi) AND aboveTrend[t]   # 30, trend up
        exit_ = (RSI[t] > exit_rsi)                       # 50
        if position == 0 and enter:
            position = 1
        else if position == 1 and exit_:
            position = 0
        # otherwise hold
    rawPos[t] = position              # desired position as of close[t]
```

---

## 5. Execution (NO lookahead) — critical

The position decided from `close[t]` is entered at the **next** bar's close:

```
heldPos[t] = rawPos[t-1]      # shift by one day
```

For **live trading**: after today's close, compute `rawPos[today]`; you hold that
position starting at the next session. The live rebalancer (`runners/daily_rebalance.py`)
does exactly this — compute signal on data through the prior close, trade today.

---

## 6. Portfolio & P&L (equal-weight, daily rebalance)

- Capital split equally across the N=6 names: each long sleeve targets `equity/N`.
- A name flat that day holds cash (0 return).
- Daily portfolio return = average across sleeves of `heldPos * dailyReturn - cost`,
  where `dailyReturn[t] = close[t]/close[t-1] - 1`.
- **Cost:** charge `3 bps` whenever a sleeve's position changes (0→1 entry or
  1→0 exit); a full round trip therefore pays 6 bps.

```
sleeveRet[t] = heldPos[t]*dailyReturn[t] - abs(heldPos[t]-heldPos[t-1]) * 0.0003
portRet[t]   = mean over sleeves of sleeveRet[t]
equity[t]    = 100_000 * prod(1 + portRet[0..t])
```

Sharpe (reporting): `mean(portRet)/std(portRet) * sqrt(252)`.

---

## 7. Data requirements

- **Daily OHLC**, split/dividend-**adjusted** (we use `auto_adjust` from the
  source). Unadjusted prices will create fake gaps at splits and corrupt signals.
- At least `trend_sma + a few` bars of history before the first tradable day
  (i.e., ≥ ~110 daily bars) per ticker.

---

## 8. Go port checklist

1. Fetch adjusted daily bars per ticker (your data feed), sorted ascending.
2. Compute `RSI[]` (recursive, alpha=0.5) and `SMA100[]`.
3. Run the state machine → `rawPos[]`; shift one day → `heldPos[]`.
4. For live: only the **last** value matters — `rawPos[lastClose]` tells you
   whether to hold each name tomorrow. Size each held name to `equity/6`,
   diff against current broker positions, send the deltas.
5. Round-trip cost 6 bps is for backtest accounting; live cost is your real
   commission/slippage.

A correct port should match these per-name trade counts over 2016–2025 with the
tuned params: **SPY 188, QQQ 186, GLD 173, MSFT 182, JPM 165, GOOGL 165** —
1,059 total, ≈ **109.5 trades/year** for the 6-name book.
