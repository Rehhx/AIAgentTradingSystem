# quant-agent

An 8-agent orchestration system for autonomous quant strategy research, validation, and paper trading. Agents propose strategies, the framework backtests them with embedding-based regime filtering, walk-forward validates them, and (if they pass risk checks) executes them on Alpaca paper accounts.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          orchestrator.py                           │
│      routes tasks, manages strategy lifecycle, persists state      │
└────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        │                         │                          │
    research layer           validation layer            execution layer
        │                         │                          │
  ┌─────┴─────┐             ┌─────┴─────┐               ┌────┴─────┐
  research    autonomous    backtesting  risk           code        execution
  ml_research                                                       monitor
```

### Agents

| layer       | agent                | status        | role                                                                  |
|-------------|----------------------|---------------|-----------------------------------------------------------------------|
| research    | `research_agent`     | scaffold      | scans arxiv/blogs via Claude Agent SDK + web_search                   |
| research    | `autonomous_agent`   | scaffold      | invents strategies from first principles (no lookups)                 |
| research    | `ml_research_agent`  | **built**     | XGBoost on engineered features (RSI, ATR, EMAs, regime one-hot, ToD)  |
| validation  | `backtesting_agent`  | **built**     | 6 strategies, regime-gated, embedding quality gate, walk-forward      |
| validation  | `risk_agent`         | **built**     | gates strategies on Sharpe / DD / WR / trade-count from `config.RISK` |
| execution   | `code_agent`         | scaffold      | turns approved specs into runnable strategy modules                   |
| execution   | `execution_agent`    | **built**     | Alpaca paper order submission (simulated fallback if creds missing)   |
| execution   | `monitor_agent`      | **built**     | polls Alpaca positions; flags drawdown/concentration breaches         |

### Vector stores

Three ChromaDB collections (`vector_stores/chroma_db/`):

- **regime_store** — 700K embedded 60-bar candle windows. `find_similar()` returns forward return distribution and regime label for any current window.
- **strategy_store** — embedded strategy specs + results, for deduplication and outcome lookup.
- **research_store** — embedded papers/docs, seeded with 12 known strategies.

Embeddings are cached locally in `vector_stores/.cache/` (sharded by sha256) so identical text is never re-embedded.

## Strategy lifecycle

```
proposed → backtesting → risk_review → approved → implementing → paper_trading
                                            ↓                         ↓
                                        rejected                    paused/retired
```

State is persisted to `results/store.json` after every transition.

## Backtesting framework

The validation pipeline applies, in order:

1. **Signal generation** — strategy-specific (`signals_rsi_reversion`, `signals_orb`, etc.).
2. **1-bar shift** — kills same-bar lookahead.
3. **Local regime gate** — `regime_label_series()` classifies each bar; only fire when strategy is in a compatible regime (`STRATEGY_REGIME_AFFINITY`).
4. **Embedding quality gate** (optional) — `precompute_regime_quality()` looks up the k=20 most similar historical 60-bar patterns from the regime store; entries require `fwd_pct_positive ≥ threshold`.
5. **ATR stop loss** — per-strategy `stop_atr_mult` (tight for trend, wide for mean-reversion).
6. **Re-entry cooldown** — 5 bars between exit and next entry.
7. **Mark-to-market equity** — tracked every bar (not just at exits).
8. **Daily Sharpe** — equity resampled to daily, annualized by √252.

## Current state

Latest run (`results/backtest_results.json`):

| strategy        | gated sharpe | drawdown | win rate | trades |
|-----------------|--------------|----------|----------|--------|
| bb_squeeze      | -1.50        | -2.1%    | 41.9%    | 908    |
| orb             | -4.81        | -22.9%   | 21.1%    | 9,908  |
| ema_crossover   | -9.78        | -47.5%   | 27.1%    | 24,239 |
| momentum        | -12.49       | -45.6%   | 31.1%    | 22,196 |

`rsi_reversion` and `vwap_reversion` are marked `active: False` — see `STRATEGIES` in `agents/backtesting_agent.py`.

The strongest signal: `bb_squeeze × NVDA` at gated Sharpe **-0.11** with 136 trades. Walk-forwarding the gate threshold confirmed the result does not generalize OOS yet (test Sharpe -0.91, overfit gap +0.92) — needs more signal work before it advances to `risk_review`.

## Setup

```bash
# 1. python 3.13 + venv
python -m venv .venv
source .venv/Scripts/activate     # bash
# or: .venv\Scripts\Activate.ps1   # powershell

# 2. install
pip install -r requirements.txt

# 3. credentials — copy and fill in
cp env.example .env
# edit .env to add OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_API_SECRET, ANTHROPIC_API_KEY

# 4. point to your parquet data dir (default: C:\Users\pcagm\Downloads\StockData)
# either edit config.DATA_DIR or set DATA_DIR=... in .env
```

The parquet files must contain columns: `symbol, Interval, EventAt, Open, High, Low, Close, Volume, Source, AggCount` — Alpaca's 1-minute bar export format.

## Running

```bash
# full backtest sweep across all active strategies × 5 tickers
python agents/backtesting_agent.py

# parameter optimization (walk-forward, train/test split per strategy)
python agents/walk_forward_runner.py

# ML model training (XGBoost on NVDA, evaluates predictions through backtest engine)
python agents/ml_research_agent.py NVDA xgboost

# embedding-based regime quality gate (precomputes + runs gated backtest)
python agents/embedding_gate_runner.py SPY bb_squeeze

# 4-strategy x 5-ticker grid of gated Sharpes (reuses cached embeddings)
python agents/strategy_ticker_grid.py

# walk-forward the embedding gate threshold itself
python agents/walk_forward_gate_runner.py bb_squeeze NVDA

# risk gate smoke test (no live data needed)
python agents/risk_agent.py

# execution / monitor agents (simulated unless ALPACA creds present)
python agents/execution_agent.py
python agents/monitor_agent.py
```

## Layout

```
agents/                  agent implementations
  backtesting_agent.py     6 strategies + engine + walk-forward + embedding gate
  ml_research_agent.py     XGBoost model trainer
  risk_agent.py            gate-or-reject on backtest metrics
  execution_agent.py       Alpaca paper order submission
  monitor_agent.py         live position health checks
  research_agent.py        Claude Agent SDK scaffold (web search)
  autonomous_agent.py      Claude Agent SDK scaffold (pure reasoning)
  code_agent.py            Claude Agent SDK scaffold (strategy code generation)
  walk_forward_runner.py        runner: grid-search strategy params
  embedding_gate_runner.py      runner: precompute + gated backtest
  walk_forward_gate_runner.py   runner: walk-forward the gate threshold
  strategy_ticker_grid.py       runner: strategies × tickers grid
data/
  loader.py                parquet loader + session filtering + resampling
vector_stores/
  client.py                OpenAI embeddings client + disk cache
  regime_store.py          60-bar pattern embeddings; find_similar()
  strategy_store.py        strategy spec embeddings; deduplication
  research_store.py        paper/doc embeddings; seeded with 12 known strategies
results/
  store.json               persistent state (strategies + tasks + trades)
  backtest_results.json    latest standalone backtest aggregates
  walk_forward_results.json  per-strategy best params + train/test Sharpe
  strategy_ticker_grid.json  grid of gated Sharpes
config.py                  paths, API keys, risk thresholds, defaults
orchestrator.py            routes tasks, manages lifecycle, persists state
requirements.txt           pinned dependencies
env.example                template for .env
```

## What "done" looks like

A strategy reaches `paper_trading` only when:

1. backtest run produced Sharpe ≥ 0.8, max DD ≥ -15%, WR ≥ 45%, ≥50 trades (`config.RISK`)
2. walk-forward train/test confirmed those numbers OOS without significant overfit gap
3. risk_agent stamped `passed: True` (no failures, warnings tracked)
4. code_agent wrote a self-contained signals module that round-trips through validation
5. execution_agent successfully placed paper orders for it
6. monitor_agent confirms positions match expected sizing and no drawdown breach

None of the 6 current rule-based strategies have cleared step 1 yet — the framework is honest about that. The lift work is in signal quality, not engine plumbing.

## License

MIT (or whatever you choose — placeholder).
