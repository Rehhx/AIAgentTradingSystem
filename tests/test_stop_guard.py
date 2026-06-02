"""Stop-guard breach logic + the bad-data sanity floor (the BNY glitch we caught).

Uses a fake agent so nothing touches Alpaca and do_liquidate=False so no orders
are ever attempted — we assert on the decision log only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import runners.stop_guard as sg


class _Clock:
    is_open = True
    timestamp = None


class _Client:
    def get_clock(self):
        return _Clock()


class _FakeAgent:
    """Minimal stand-in for ExecutionAgent: just positions + a clock."""
    simulated = False

    def __init__(self, positions):
        self._pos = positions
        self.client = _Client()

    def get_positions(self):
        return self._pos


def _pos(symbol, qty, entry, current):
    return {"symbol": symbol, "qty": qty, "avg_entry_price": entry,
            "market_value": qty * current, "unrealized_pl": qty * (current - entry)}


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    # never read/write the real results/position_highs.json during tests
    monkeypatch.setattr(sg, "STATE", tmp_path / "highs.json")


def _log_for(agent, **kw):
    return enforce_join(sg.enforce_stops(agent, do_liquidate=False, **kw))


def enforce_join(lines):
    return "\n".join(lines)


def test_bad_data_is_suspect_not_sold():
    # BNY: entry 139.77, glitched current 10.44 (-92%) -> must be SUSPECT, never sold
    agent = _FakeAgent([_pos("BNY", 2.25, 139.77, 10.44)])
    log = _log_for(agent)
    assert "SUSPECT BNY" in log
    assert "LIQUIDATE" not in log


def test_real_breach_flagged():
    # WSM: entry 207.26, current 176.16 (-15%) -> real hard-stop breach
    agent = _FakeAgent([_pos("WSM", 1.64, 207.26, 176.16)])
    log = _log_for(agent, hard_pct=15.0)
    assert "BREACH WSM" in log
    assert "SUSPECT" not in log


def test_healthy_position_not_flagged():
    agent = _FakeAgent([_pos("AAPL", 22.0, 312.0, 306.0)])    # -2%, fine
    log = _log_for(agent)
    assert "AAPL" not in log or "none breached" in log
    assert "BREACH" not in log and "SUSPECT" not in log


def test_shorts_are_skipped():
    agent = _FakeAgent([_pos("TLT", -73.0, 85.0, 95.0)])      # short, not guarded here
    log = _log_for(agent)
    assert "TLT" not in log


def test_trailing_stop_triggers_off_high_water():
    # bought at 100, ran to 200 (high-water), now 150 = -25% off the high -> trailing breach
    sg._save({})  # start clean (autouse fixture points STATE at tmp)
    agent = _FakeAgent([_pos("RUN", 10.0, 100.0, 200.0)])
    sg.enforce_stops(agent, trail_pct=20.0, do_liquidate=False)   # seed high-water at 200
    agent2 = _FakeAgent([_pos("RUN", 10.0, 100.0, 150.0)])
    log = enforce_join(sg.enforce_stops(agent2, trail_pct=20.0, hard_pct=50.0, do_liquidate=False))
    assert "BREACH RUN" in log and "trailing" in log
