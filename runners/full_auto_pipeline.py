"""
agents/full_auto_pipeline.py
----------------------------
end-to-end automated research→test→validate→risk pipeline. one entrypoint
that wires together every agent in the system so the user can press a
single button and get a ranked verdict.

flow:
  1a. RESEARCH      — research_agent finds known + invented strategies
                      (registry-aware; won't propose duplicates)
  1b. AUTONOMOUS    — autonomous_agent invents from first principles
                      (no web tools, also registry-aware)
  1c. ML_RESEARCH   — ml_research_agent.research() proposes ML approaches
                      (architecture + features + target combos)
  2.  MATCH         — each rule-based idea is mapped to a registered
                      STRATEGIES key (or flagged for code_agent)
  3.  BACKTEST      — every matched strategy runs on the universe with
                      dump_trades=True so the user can audit
  4.  WALK-FORWARD  — strategies with backtest Sharpe > -1 go through
                      70/30 walk-forward — exposes overfitting
  5.  RISK          — risk_agent.evaluate() applies config.RISK thresholds
  6.  REPORT        — single ranked JSON with per-agent sections

agent separation of concerns:
  research_agent      — discovers known strategies AND invents novel ones,
                        knows what's already in the registry
  autonomous_agent    — pure first-principles invention, no web access
  ml_research_agent   — TWO modes: research() proposes ML approaches,
                        run() trains and evaluates them
  code_agent          — turns unmatched idea specs into signal() functions
                        (flagged but NOT auto-invoked here — would burn tokens)
  backtesting_agent   — validates implementations
  risk_agent          — gate on Sharpe / DD / WR / trade-count

usage:
    python agents/full_auto_pipeline.py
    python agents/full_auto_pipeline.py --refresh-research --quick
    python agents/full_auto_pipeline.py --strategies bb_squeeze,noise_area_breakout
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from agents.backtesting_agent import (
    BacktestingAgent, STRATEGIES, walk_forward_optimize,
)
from agents.risk_agent      import RiskAgent
from data.loader            import DATA_DIR

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("auto_pipeline")
log.setLevel(logging.INFO)


# WF grids per strategy — keep small for speed. only sweep the params that
# actually matter for each strategy's edge.
WF_GRIDS = {
    "bb_squeeze":            {"bb_period": [15, 25], "bb_std": [2.0, 2.5], "kc_mult": [1.5, 2.0]},
    "noise_area_breakout":   {"sigma_mult": [0.5, 1.0, 1.5], "lookback_days": [14, 21]},
    "overnight_gap_fade":    {"gap_threshold_pct": [0.005, 0.010, 0.015]},
    "extreme_bar_fade":      {"body_atr_mult": [2.0, 3.0, 4.0], "hold_bars": [5, 10, 20]},
    "trend_ride":            {"ema_period": [30, 50, 80], "breakout_lookback": [20, 30, 50]},
    "vwap_slope_break":      {"slope_lookback_bars": [5, 10, 20]},
    "half_hour_continuation":{"threshold_bps": [5, 10, 25], "lookback_days": [20, 40]},
    "bb_band_touch_revert_v2":{"rsi_overbought": [70, 75], "rsi_oversold": [25, 30]},
}


def _norm(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _match_to_strategy(idea_name: str) -> str | None:
    """fuzzy-match a research idea to a registered STRATEGIES key."""
    norm = _norm(idea_name)
    # exact
    if norm in STRATEGIES:
        return norm
    # substring either direction
    for key in STRATEGIES:
        if key in norm or norm in key:
            return key
    # token overlap (>=2 common tokens)
    norm_tokens = set(norm.split("_"))
    for key in STRATEGIES:
        if len(norm_tokens & set(key.split("_"))) >= 2:
            return key
    return None


def phase_research(refresh: bool) -> list:
    """phase 1a — research_agent: discovery + invention, registry-aware."""
    round2 = Path("results/research_ideas_round2.json")
    round1 = Path("results/research_ideas.json")
    cached = None
    for p in (round2, round1):
        if p.exists():
            cached = p
            break

    if refresh or cached is None:
        log.info("calling research_agent via Claude SDK (this may take 60-90s)…")
        from agents.research_agent import ResearchAgent
        agent = ResearchAgent()
        result = agent.run({"payload": {"query": (
            "Find intraday US equity strategies for 1-5 minute bars. Mix of "
            "published research (cite source) and your own inventions."
        )}})
        if not result.get("success"):
            log.warning(f"research SDK call failed: {result.get('reason')}")
            return []
        return result.get("strategies_found") or []

    log.info(f"using cached research ideas from {cached}")
    with open(cached) as f:
        data = json.load(f)
    return data.get("strategies", [])


def phase_autonomous(refresh: bool) -> list:
    """phase 1b — autonomous_agent: pure first-principles invention."""
    cached = Path("results/autonomous_ideas.json")
    if not refresh and cached.exists():
        log.info(f"using cached autonomous ideas from {cached}")
        with open(cached) as f:
            return json.load(f).get("ideas", [])

    log.info("calling autonomous_agent via Claude SDK (no web tools)…")
    try:
        from agents.autonomous_agent import AutonomousAgent
        agent  = AutonomousAgent()
        result = agent.run({"payload": {"seed": (
            "Invent 3 novel intraday strategies that exploit microstructure "
            "patterns NOT covered by our existing registry."
        )}})
        ideas = result.get("ideas") or []
        cached.parent.mkdir(parents=True, exist_ok=True)
        with open(cached, "w", encoding="utf-8") as f:
            json.dump({"ideas": ideas,
                       "run_at": datetime.now(timezone.utc).isoformat()},
                      f, indent=2, ensure_ascii=False, default=str)
        log.info(f"autonomous_agent returned {len(ideas)} ideas (cached to {cached})")
        return ideas
    except Exception as e:
        log.warning(f"autonomous_agent failed: {e}")
        return []


def phase_options_research(refresh: bool) -> list:
    """phase 1d — options_research_agent: options-specific strategy ideas."""
    cached = Path("results/options_ideas.json")
    if not refresh and cached.exists():
        log.info(f"using cached options ideas from {cached}")
        with open(cached) as f:
            return json.load(f).get("ideas", [])

    log.info("calling options_research_agent via Claude SDK…")
    try:
        from agents.options_research_agent import OptionsResearchAgent
        agent  = OptionsResearchAgent()
        result = agent.run({"payload": {"query": (
            "Find SPY/QQQ options strategies tradeable via Alpaca paper, "
            "with documented edge or novel structural mechanism."
        )}})
        if not result.get("success"):
            log.warning(f"options_research failed: {result.get('reason')}")
            return []
        ideas = result.get("ideas") or []
        cached.parent.mkdir(parents=True, exist_ok=True)
        with open(cached, "w", encoding="utf-8") as f:
            json.dump({"ideas": ideas,
                       "run_at": datetime.now(timezone.utc).isoformat()},
                      f, indent=2, default=str, ensure_ascii=False)
        log.info(f"options_research_agent returned {len(ideas)} ideas (cached to {cached})")
        return ideas
    except Exception as e:
        log.warning(f"options_research_agent failed: {e}")
        return []


def phase_implement_options(ideas: list) -> dict:
    """phase 2d — options_code_agent: generate options strategy modules."""
    from agents.options_code_agent import OptionsCodeAgent
    agent = OptionsCodeAgent()
    generated, dupes, failed = [], [], []
    for idea in ideas:
        name = (idea.get("name") or "").lower().strip().replace(" ", "_")
        if not name:
            continue
        if _match_to_strategy(name) is not None:
            log.info(f"  options_code_agent: skip '{name}' — duplicate")
            dupes.append(name)
            continue
        log.info(f"  options_code_agent: implementing '{name}' "
                 f"(structure={idea.get('structure', '?')})…")
        result = agent.generate_from_spec(idea)
        if result.get("success"):
            generated.append(result["name"])
            log.info(f"    -> registered at {result['code_path']}")
        elif result.get("duplicate"):
            dupes.append(name)
        else:
            failed.append({"name": name, "reason": result.get("reason")})
            log.warning(f"    -> FAILED: {result.get('reason')}")
    return {"generated": generated, "skipped_duplicate": dupes, "failed": failed}


def phase_ml_research(refresh: bool) -> list:
    """phase 1c — ml_research_agent.research(): ML approach proposals."""
    cached = Path("results/ml_research_approaches.json")
    if not refresh and cached.exists():
        log.info(f"using cached ml_research approaches from {cached}")
        with open(cached) as f:
            return json.load(f).get("approaches", [])

    log.info("calling ml_research_agent.research() via Claude SDK…")
    try:
        from agents.ml_research_agent import MLResearchAgent
        agent  = MLResearchAgent()
        result = agent.research(query=(
            "Our XGBoost baseline AUC is ~0.51 on standard TA features. "
            "Propose ML/DL approaches that could exploit signal we're missing."
        ))
        approaches = result.get("approaches") or []
        cached.parent.mkdir(parents=True, exist_ok=True)
        with open(cached, "w", encoding="utf-8") as f:
            json.dump({"approaches": approaches,
                       "run_at": datetime.now(timezone.utc).isoformat()},
                      f, indent=2, ensure_ascii=False, default=str)
        log.info(f"ml_research_agent returned {len(approaches)} approaches (cached to {cached})")
        return approaches
    except Exception as e:
        log.warning(f"ml_research_agent failed: {e}")
        return []


def phase_implement_ml(approaches: list) -> dict:
    """
    phase 2c — ml_code_agent: turn ML approach proposals into runnable
    strategy modules. each approach has architecture/features/target spec;
    ml_code_agent generates a module that trains-on-the-fly inside signals().
    """
    from agents.ml_code_agent import MLCodeAgent
    agent = MLCodeAgent()
    generated, dupes, failed = [], [], []

    for appr in approaches:
        name = (appr.get("name") or "").lower().strip().replace(" ", "_")
        if not name:
            continue
        if _match_to_strategy(name) is not None:
            log.info(f"  ml_code_agent: skip '{name}' — matches existing strategy")
            dupes.append(name)
            continue

        log.info(f"  ml_code_agent: implementing '{name}' "
                 f"(arch={appr.get('architecture', '?')})…")
        # ml_code_agent expects the same spec shape — already has architecture,
        # features, target, hypothesis fields from ml_research_agent
        result = agent.generate_from_spec(appr)
        if result.get("success"):
            generated.append(result["name"])
            log.info(f"    -> registered at {result['code_path']}")
        elif result.get("duplicate"):
            dupes.append(name)
        else:
            failed.append({"name": name, "reason": result.get("reason")})
            log.warning(f"    -> FAILED: {result.get('reason')}")

    return {"generated": generated, "skipped_duplicate": dupes, "failed": failed}


def phase_implement(ideas: list, source: str = "autonomous") -> dict:
    """
    phase 2b — code_agent: turn unmatched ideas into runnable strategies.

    for each idea that is NOT already in STRATEGIES (by exact or substring
    match), call code_agent.generate_from_spec(). it generates a module,
    validates it on synthetic data, and registers it into STRATEGIES so
    downstream backtest can pick it up by name.

    returns {generated: [names...], skipped_duplicate: [...], failed: [...]}
    """
    from agents.code_agent import CodeAgent
    agent = CodeAgent()
    generated, dupes, failed = [], [], []

    for idea in ideas:
        name = (idea.get("name") or "").lower().strip().replace(" ", "_")
        if not name:
            continue
        # quick local dedup before burning a token
        if _match_to_strategy(name) is not None:
            log.info(f"  code_agent: skip '{name}' — matches existing strategy")
            dupes.append(name)
            continue

        log.info(f"  code_agent: implementing '{name}' (source={source})…")
        result = agent.generate_from_spec(idea)
        if result.get("success"):
            generated.append(result["name"])
            log.info(f"    -> registered at {result['code_path']}")
        elif result.get("duplicate"):
            dupes.append(name)
            log.info(f"    -> skip (Claude detected duplicate)")
        else:
            failed.append({"name": name, "reason": result.get("reason")})
            log.warning(f"    -> FAILED: {result.get('reason')}")

    return {"generated": generated, "skipped_duplicate": dupes, "failed": failed}


def phase_match(ideas: list) -> dict:
    """phase 2 — map research ideas to registered strategy keys."""
    matched   = {}     # idea_name -> strategy_key
    unmatched = []
    for idea in ideas:
        name = idea.get("name", "")
        key = _match_to_strategy(name)
        if key:
            matched[name] = key
        else:
            unmatched.append(name)

    log.info(f"matched {len(matched)}/{len(ideas)} research ideas to known strategies")
    if unmatched:
        log.info(f"  needs code_agent (unmatched): {unmatched}")
    return {"matched": matched, "unmatched": unmatched}


def phase_backtest(strategy_keys: list, tickers: list, start: str, end: str) -> dict:
    """phase 3 — backtest each strategy on the universe."""
    agent = BacktestingAgent(data_dir=DATA_DIR)
    out = {}
    for key in strategy_keys:
        log.info(f"  backtesting {key} on {len(tickers)} tickers…")
        r = agent.run({"payload": {
            "name": key, "tickers": tickers, "start": start, "end": end,
            "dump_trades": True,
        }})
        if not r.get("success"):
            log.warning(f"  {key} backtest FAILED: {r.get('reason')}")
            continue
        agg = r["aggregate"]
        out[key] = {
            "aggregate":  agg,
            "per_ticker": {t: {k: rt[k] for k in ("sharpe", "max_drawdown", "win_rate", "total_trades")}
                           for t, rt in r["per_ticker"].items()},
        }
        log.info(f"  {key} | sharpe={agg['sharpe']:.2f} dd={agg['max_drawdown']:.2%} "
                 f"wr={agg['win_rate']:.2%} trades={agg['total_trades']}")
    return out


def phase_walk_forward(strategy_keys: list, tickers: list, start: str, end: str) -> dict:
    """phase 4 — walk-forward each strategy's grid; surface overfitting."""
    out = {}
    for key in strategy_keys:
        grid = WF_GRIDS.get(key)
        if not grid:
            log.info(f"  no WF grid defined for {key} — skipping walk-forward")
            continue
        log.info(f"  walk-forwarding {key} (grid={list(grid.keys())})…")
        r = walk_forward_optimize(
            strategy_name = key,
            param_grid    = grid,
            tickers       = tickers,
            start         = start,
            end           = end,
            train_pct     = 0.7,
        )
        if not r.get("success"):
            log.warning(f"  {key} WF FAILED: {r.get('reason')}")
            continue
        out[key] = {
            "best_params":  r["best_params"],
            "train_sharpe": r["train_sharpe"],
            "test_sharpe":  r["test_sharpe"],
            "overfit_gap":  r["overfit_gap"],
        }
        verdict = "PASS" if r["test_sharpe"] > 0 else "FAIL"
        log.info(f"  {key} WF | train={r['train_sharpe']:.2f} test={r['test_sharpe']:.2f} "
                 f"gap={r['overfit_gap']:.2f} [{verdict}]")
    return out


def phase_champions(backtest_results: dict, wf_results: dict, risk_results: dict,
                    n_keep: int = 8, path: Path = Path("results/champions.json")) -> dict:
    """
    phase 5b — keep the best strategies across runs.

    persistent "hall of fame" — every run merges its results with the existing
    champions file, sorts by a stable score (test sharpe if WF available, else
    backtest sharpe), and keeps the top N. underperformers are moved to a
    "retired_this_run" list in the same file so they're discoverable but no
    longer crowd the champion slots.

    a strategy "earns" champion status by being in the top N at any point.
    once a new generation of strategies arrives that outperforms it, it cycles
    out — that's the evolutionary part the user asked for.
    """
    now = datetime.now(timezone.utc).isoformat()

    # build candidates from the current run
    candidates = []
    for name, r in backtest_results.items():
        agg   = r["aggregate"]
        wf    = wf_results.get(name) or {}
        risk_v = risk_results.get(name) or {}
        candidates.append({
            "name":             name,
            "sharpe":           agg["sharpe"],
            "max_drawdown":     agg["max_drawdown"],
            "win_rate":         agg["win_rate"],
            "total_trades":     agg["total_trades"],
            "pnl_dollars":      agg.get("pnl_dollars"),
            "final_capital":    agg.get("final_capital"),
            "total_return":     agg.get("total_return"),
            "test_sharpe":      wf.get("test_sharpe"),
            "overfit_gap":      wf.get("overfit_gap"),
            "risk_passed":      risk_v.get("passed", False),
            "first_seen":       now,
            "last_run":         now,
        })

    # merge with existing champions — preserve first_seen
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text()).get("champions", [])
        except Exception:
            existing = []

    by_name = {c["name"]: c for c in existing}
    for c in candidates:
        if c["name"] in by_name:
            c["first_seen"] = by_name[c["name"]]["first_seen"]
        by_name[c["name"]] = c

    # rank by best available score:
    #   primary: test_sharpe (proves it survived walk-forward)
    #   fallback: backtest sharpe (no WF run yet)
    def _score(c):
        ts = c.get("test_sharpe")
        return ts if ts is not None else c["sharpe"]

    ranked   = sorted(by_name.values(), key=_score, reverse=True)
    champions = ranked[:n_keep]
    retired   = ranked[n_keep:]

    # what changed this run?
    prev_names = {c["name"] for c in existing}
    new_champs    = [c["name"] for c in champions if c["name"] not in prev_names]
    dropped       = [c["name"] for c in retired   if c["name"] in prev_names]
    promoted_from_retired = []  # strategies that were retired and came back

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "champions":        champions,
            "retired_this_run": retired,
            "new_this_run":     new_champs,
            "dropped_this_run": dropped,
            "n_keep":           n_keep,
            "scored_by":        "test_sharpe (with backtest sharpe fallback)",
            "updated_at":       now,
        }, f, indent=2, default=str, ensure_ascii=False)

    return {
        "champions":  champions,
        "retired":    retired,
        "new":        new_champs,
        "dropped":    dropped,
    }


def phase_risk(backtest_results: dict) -> dict:
    """phase 5 — risk_agent.evaluate against config.RISK thresholds."""
    risk = RiskAgent(store=None)
    out  = {}
    for key, res in backtest_results.items():
        agg = res["aggregate"]
        # risk_agent expects a flat dict
        check = risk.evaluate({
            **agg,
            "per_ticker": res["per_ticker"],
        })
        out[key] = check
        verdict = "PASS" if check["passed"] else "FAIL"
        log.info(f"  {key} risk | {verdict} | failures={check['failures']}")
    return out


def phase_report(matched: dict, backtest: dict, wf: dict, risk: dict, path: Path,
                 autonomous_ideas: Optional[list] = None,
                 ml_approaches:    Optional[list] = None) -> dict:
    """phase 6 — single JSON report with per-agent sections."""
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "agents": {
            "research_agent":     {"matched": matched["matched"],
                                   "unmatched": matched["unmatched"]},
            "autonomous_agent":   {"ideas": autonomous_ideas or []},
            "ml_research_agent":  {"approaches": ml_approaches or []},
            "backtesting_agent":  backtest,
            "risk_agent":         risk,
        },
        "walk_forward":    wf,
        "ranking_by_test_sharpe": sorted(
            [(k, v["test_sharpe"]) for k, v in wf.items()],
            key=lambda x: x[1], reverse=True,
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-research", action="store_true",
                    help="call research_agent for new ideas (costs tokens)")
    ap.add_argument("--quick", action="store_true",
                    help="smaller universe + shorter period for fast iteration")
    ap.add_argument("--strategies", default=None,
                    help="comma-separated subset of strategy keys (skips research/match)")
    ap.add_argument("--no-wf", action="store_true",
                    help="skip walk-forward (just backtest + risk)")
    ap.add_argument("--implement-autonomous", action="store_true",
                    help="invoke code_agent on unmatched autonomous + research "
                         "ideas (burns ~1 SDK call per novel idea)")
    ap.add_argument("--implement-ml", action="store_true",
                    help="invoke ml_code_agent on ML approach proposals from "
                         "ml_research_agent (burns ~1 SDK call per proposal)")
    ap.add_argument("--implement-options", action="store_true",
                    help="invoke options_research_agent + options_code_agent "
                         "to generate options strategies (4-6 ideas + impl)")
    ap.add_argument("--tickers", default=None,
                    help="comma-separated ticker list, or 'all' for every "
                         "parquet in DATA_DIR. overrides --quick default.")
    args = ap.parse_args()

    # register the daily (multi-day hold) strategies so they're selectable via
    # --strategies daily_rsi2_meanrev / daily_donchian / daily_trend_5020 and
    # are included in the "active registry" sweep. NOTE: the authoritative
    # board numbers for these come from runners/daily_book.py (portfolio-level,
    # full history); the per-ticker pipeline run is for risk_agent coverage.
    try:
        from agents.daily_strategies import register_daily_strategies
        register_daily_strategies()
    except Exception as e:
        log.warning(f"could not register daily strategies: {e}")

    if args.tickers:
        if args.tickers.lower() == "all":
            from data.loader import available_tickers
            tickers = available_tickers()
            log.info(f"using ALL {len(tickers)} tickers from DATA_DIR")
        else:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = ["SPY", "AAPL", "MSFT"] if args.quick else ["SPY", "QQQ", "AAPL", "MSFT", "CAT"]
    start   = "2023-01-01" if args.quick else "2022-01-01"
    end     = "2025-01-01"

    print(f"\n{'='*72}")
    print(f"  FULL AUTO PIPELINE")
    print(f"  tickers={tickers}  range={start}..{end}  quick={args.quick}")
    print(f"{'='*72}\n")

    # PHASE 1+2 — pick strategies to test
    autonomous_ideas, ml_approaches, options_ideas = [], [], []
    implement_info         = {"generated": [], "skipped_duplicate": [], "failed": []}
    ml_implement_info      = {"generated": [], "skipped_duplicate": [], "failed": []}
    options_implement_info = {"generated": [], "skipped_duplicate": [], "failed": []}
    if args.strategies:
        keys = [k.strip() for k in args.strategies.split(",") if k.strip() in STRATEGIES]
        log.info(f"PHASE 1+2: user-specified subset: {keys}")
        match_info = {"matched": {k: k for k in keys}, "unmatched": []}
    else:
        log.info("PHASE 1a: research_agent (registry-aware discovery + invention)")
        research_ideas = phase_research(refresh=args.refresh_research)

        log.info("PHASE 1b: autonomous_agent (first-principles invention)")
        autonomous_ideas = phase_autonomous(refresh=args.refresh_research)

        log.info("PHASE 1c: ml_research_agent.research() (ML approach proposals)")
        ml_approaches = phase_ml_research(refresh=args.refresh_research)

        if args.implement_options:
            log.info("PHASE 1d: options_research_agent (options-specific ideas)")
            options_ideas = phase_options_research(refresh=args.refresh_research)

        # for matching we use research_ideas — they have name fields that map
        # cleanly. autonomous ideas + ml approaches are recorded but require
        # code_agent / training-pipeline integration before they can backtest.
        log.info(f"PHASE 2: matching {len(research_ideas)} research ideas to registry")
        match_info = phase_match(research_ideas)
        # surface autonomous ideas that don't overlap registry as "unmatched"
        # so the report knows code_agent is needed
        for idea in autonomous_ideas:
            name = idea.get("name", "")
            if name and _match_to_strategy(name) is None:
                match_info["unmatched"].append(f"autonomous:{name}")
        for appr in ml_approaches:
            name = appr.get("name", "")
            if name:
                match_info["unmatched"].append(f"ml_research:{name}")

        keys = sorted(set(match_info["matched"].values()))

        # PHASE 2b — code_agent (opt-in): generate signal modules for
        # unmatched autonomous AND unmatched research ideas. each successful
        # generation registers a new key in STRATEGIES so it joins the keys
        # list and gets backtested in PHASE 3.
        if args.implement_autonomous:
            implement_candidates = list(autonomous_ideas)
            # also pull research ideas that didn't match anything in registry —
            # research_agent often produces inventions (kind=invention) that
            # require code_agent before they can be backtested.
            unmatched_research_names = {
                _norm(n) for n in match_info["unmatched"]
                if not n.startswith(("autonomous:", "ml_research:"))
            }
            for idea in research_ideas:
                if _norm(idea.get("name") or "") in unmatched_research_names:
                    implement_candidates.append(idea)

            if implement_candidates:
                log.info(f"PHASE 2b: code_agent implementing "
                         f"{len(autonomous_ideas)} autonomous + "
                         f"{len(implement_candidates) - len(autonomous_ideas)} research ideas")
                implement_info = phase_implement(implement_candidates, source="mixed")
                keys.extend(implement_info["generated"])
                keys = sorted(set(keys))

        # PHASE 2c — ml_code_agent: turn ml_research_agent's ML approach
        # proposals into trainable+predicting strategy modules.
        if args.implement_ml and ml_approaches:
            log.info(f"PHASE 2c: ml_code_agent implementing {len(ml_approaches)} ML approaches")
            ml_implement_info = phase_implement_ml(ml_approaches)
            keys.extend(ml_implement_info["generated"])
            keys = sorted(set(keys))

        # PHASE 2d — options_code_agent: implement options strategies. their
        # signals() runs the underlying signal through the standard backtest,
        # and they also export options_intent() for options_agent execution.
        if args.implement_options and options_ideas:
            log.info(f"PHASE 2d: options_code_agent implementing "
                     f"{len(options_ideas)} options strategies")
            options_implement_info = phase_implement_options(options_ideas)
            keys.extend(options_implement_info["generated"])
            keys = sorted(set(keys))

        # also include active registry strategies for full coverage
        for key, entry in STRATEGIES.items():
            if entry[1].get("active") and key not in keys and len(keys) < 10:
                keys.append(key)

    if not keys:
        log.warning("no strategies to test — exiting")
        return
    log.info(f"will test: {keys}")

    # PHASE 3 — backtest
    log.info(f"PHASE 3: backtesting {len(keys)} strategies")
    bt = phase_backtest(keys, tickers, start, end)

    # PHASE 4 — walk-forward
    wf = {}
    if not args.no_wf:
        log.info(f"PHASE 4: walk-forward")
        # only WF strategies that posted Sharpe > -1 on backtest — saves time
        wf_keys = [k for k, r in bt.items() if r["aggregate"]["sharpe"] > -1.0]
        if not wf_keys:
            wf_keys = list(bt.keys())[:3]  # at least try the top 3
        log.info(f"  walk-forward candidates: {wf_keys}")
        wf = phase_walk_forward(wf_keys, tickers, start, end)

    # PHASE 5 — risk
    log.info(f"PHASE 5: risk_agent")
    risk = phase_risk(bt)

    # PHASE 5b — champion cycling
    log.info(f"PHASE 5b: champion cycling (keep top 8 by test sharpe)")
    champ_info = phase_champions(bt, wf, risk, n_keep=8)
    log.info(f"  champions: {len(champ_info['champions'])} (new this run: {champ_info['new']}, "
             f"dropped: {champ_info['dropped']})")

    # PHASE 6 — report
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path  = Path(f"results/auto_pipeline_{stamp}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = phase_report(match_info, bt, wf, risk, path,
                          autonomous_ideas=autonomous_ideas,
                          ml_approaches=ml_approaches)

    # console summary
    print(f"\n{'='*72}\n  AUTO PIPELINE SUMMARY\n{'='*72}")
    print(f"\n  backtest ranking by sharpe (PnL from $100k starting balance):")
    ranked = sorted(bt.items(), key=lambda kv: kv[1]["aggregate"]["sharpe"], reverse=True)
    for i, (k, r) in enumerate(ranked, 1):
        agg = r["aggregate"]
        risk_verdict = "PASS" if risk[k]["passed"] else "FAIL"
        pnl   = agg.get("pnl_dollars", 0.0)
        final = agg.get("final_capital", 100000)
        pnl_sign = "+" if pnl >= 0 else "-"
        print(f"  {i}. {k:<26} sharpe={agg['sharpe']:>6.2f}  PnL={pnl_sign}${abs(pnl):>9,.0f}  "
              f"final=${final:>10,.0f}  dd={agg['max_drawdown']:>7.2%}  "
              f"wr={agg['win_rate']:>5.1%}  trades={agg['total_trades']:>5}  [{risk_verdict}]")

    if wf:
        print(f"\n  walk-forward (overfit detector):")
        wf_ranked = sorted(wf.items(), key=lambda kv: kv[1]["test_sharpe"], reverse=True)
        for k, w in wf_ranked:
            verdict = "PASS" if w["test_sharpe"] > 0 else "FAIL"
            print(f"    {k:<26} train={w['train_sharpe']:>6.2f}  test={w['test_sharpe']:>6.2f}  "
                  f"gap={w['overfit_gap']:>6.2f}  [{verdict}]")

    if autonomous_ideas:
        gen_names = set(implement_info.get("generated", []))
        print(f"\n  autonomous_agent ideas:")
        for idea in autonomous_ideas:
            n = (idea.get("name") or "").lower().replace(" ", "_")
            mark = "[IMPLEMENTED]" if n in gen_names else \
                   "[SKIPPED-DUP]" if n in implement_info.get("skipped_duplicate", []) else \
                   "[needs code_agent]"
            print(f"    {mark} {idea.get('name', '?')}: {idea.get('hypothesis', '?')[:70]}")
    if ml_approaches:
        print(f"\n  ml_research_agent ML approach proposals:")
        for appr in ml_approaches:
            print(f"    - {appr.get('name', '?')} ({appr.get('architecture', '?')})")

    # CHAMPION SUMMARY — what the system has "kept" so far
    print(f"\n{'='*72}\n  CHAMPIONS (hall of fame, top 8 by test Sharpe)\n{'='*72}")
    for i, c in enumerate(champ_info["champions"], 1):
        ts = c.get("test_sharpe")
        score = f"test={ts:>6.2f}" if ts is not None else f"sharpe={c['sharpe']:>6.2f}"
        risk_mark = "PASS" if c.get("risk_passed") else " "
        age = c.get("first_seen", "?")[:10]
        pnl = c.get("pnl_dollars") or 0
        pnl_sign = "+" if pnl >= 0 else "-"
        print(f"  {i}. {c['name']:<28} {score}  PnL={pnl_sign}${abs(pnl):>9,.0f}  "
              f"dd={c['max_drawdown']:>7.2%}  trades={c['total_trades']:>5}  "
              f"[{risk_mark}]  first_seen={age}")
    if champ_info["new"]:
        print(f"\n  NEW this run: {champ_info['new']}")
    if champ_info["dropped"]:
        print(f"  DROPPED this run: {champ_info['dropped']}")
    print(f"  hall-of-fame file: results/champions.json")

    print(f"\n  full report: {path}")
    print(f"  per-trade CSVs: results/trades/")
    print(f"\n  agent separation of concerns:")
    print(f"    research_agent     -> {len(match_info['matched'])} matched, "
          f"{len([u for u in match_info['unmatched'] if not u.startswith(('autonomous:','ml_research:'))])} unmatched")
    print(f"    autonomous_agent   -> {len(autonomous_ideas)} novel ideas")
    print(f"    code_agent         -> {len(implement_info['generated'])} implemented, "
          f"{len(implement_info['skipped_duplicate'])} skipped (dupe), "
          f"{len(implement_info['failed'])} failed")
    print(f"    ml_code_agent      -> {len(ml_implement_info['generated'])} implemented, "
          f"{len(ml_implement_info['skipped_duplicate'])} skipped (dupe), "
          f"{len(ml_implement_info['failed'])} failed")
    print(f"    options_research_agent -> {len(options_ideas)} options ideas")
    print(f"    options_code_agent -> {len(options_implement_info['generated'])} implemented, "
          f"{len(options_implement_info['skipped_duplicate'])} skipped (dupe), "
          f"{len(options_implement_info['failed'])} failed")
    print(f"    ml_research_agent  -> {len(ml_approaches)} ML approach proposals")
    print(f"    backtesting_agent  -> {len(bt)} strategies validated")
    print(f"    risk_agent         -> {sum(1 for v in risk.values() if v['passed'])} passed")


if __name__ == "__main__":
    main()
