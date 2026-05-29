"""
agents/strategy_ledger.py
-------------------------
Persistent record of EVERY strategy the system has touched — kept even after the
unprofitable strategy modules are deleted from strategies/. This is the agents'
institutional memory: it tells them which ideas have already been tried (and
failed) so they don't waste a cycle re-proposing a known-dead strategy, and
which ones are profitable/deployed so they build on them instead of duplicating.

The ledger survives file deletion on purpose: we remove the dead CODE to keep the
repo clean, but retain the KNOWLEDGE of what was tried.

Status values:
  deployed   — profitable, passes risk, in production (the daily books)
  kept       — retained in the codebase but not gate-passing (held for reference)
  dead       — tried and rejected; code removed. DO NOT re-propose.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("strategy_ledger")

LEDGER_PATH = Path("results/strategy_ledger.json")


def load_ledger() -> dict:
    if LEDGER_PATH.exists():
        try:
            return json.loads(LEDGER_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_ledger(d: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")


def record(name: str, status: str, sharpe=None, kind=None, note=None,
           source_file=None) -> dict:
    """upsert one strategy into the ledger."""
    if not name:
        return load_ledger()
    d = load_ledger()
    e = d.get(name, {"name": name})
    for k, v in dict(status=status, sharpe=sharpe, kind=kind, note=note,
                     source_file=source_file).items():
        if v is not None:
            e[k] = v
    d[name] = e
    save_ledger(d)
    return d


def ledger_summary_for_prompt(max_dead: int = 200) -> str:
    """compact text for agent prompts: what's deployed (build on) and what's
    dead (never re-propose)."""
    d = load_ledger()
    if not d:
        return "  (no prior strategies recorded)"
    deployed = [v for v in d.values() if v.get("status") in ("deployed", "kept")]
    dead     = [v for v in d.values() if v.get("status") == "dead"]
    lines = []
    if deployed:
        lines.append("  PROFITABLE / RETAINED (build on these, do not duplicate):")
        for v in sorted(deployed, key=lambda x: -(x.get("sharpe") or -9)):
            lines.append(f"    - {v['name']} (sharpe {v.get('sharpe', '?')})")
    if dead:
        lines.append("  ALREADY TRIED & REJECTED — code removed, DO NOT re-propose "
                     "these or trivial variants:")
        for v in sorted(dead, key=lambda x: (x.get("sharpe") or 0))[:max_dead]:
            sh = v.get("sharpe")
            lines.append(f"    - {v['name']}" + (f" (sharpe {sh})" if sh is not None else ""))
    return "\n".join(lines)
