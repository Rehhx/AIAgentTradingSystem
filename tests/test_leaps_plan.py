"""LEAPS book plan (account 3): defined-risk sizing in model mode (no Alpaca)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import runners.options_book as ob


@pytest.fixture(autouse=True)
def _fixed_market(monkeypatch):
    # deterministic: SPY/QQQ at 500, VIX 18% -> no network, no Alpaca
    monkeypatch.setattr(ob, "_spot_fallback", lambda sym: 500.0)
    monkeypatch.setattr(ob, "_vix", lambda: 0.18)


def test_plan_is_defined_risk_and_sized_reasonably():
    pl = ob.leaps_plan(None, 100_000.0, leverage=1.0)
    assert pl["ok"]
    # defined-risk: total premium IS the max loss, and it leaves a cash buffer
    assert pl["total_cost"] > 0
    assert pl["cash_left"] == pytest.approx(100_000.0 - pl["total_cost"], abs=1.0)
    assert pl["cash_left"] > 0                       # never spends the whole account at 1.0x
    # deep-ITM 1yr calls ~ 20-35% of notional in premium at 1.0x
    assert 0.10 < pl["pct_of_equity"] < 0.45


def test_plan_covers_both_index_underlyings():
    pl = ob.leaps_plan(None, 100_000.0, leverage=1.0)
    unds = {r["underlying"] for r in pl["rows"] if r.get("ok")}
    assert unds == {"SPY", "QQQ"}


def test_leverage_scales_premium_up():
    one = ob.leaps_plan(None, 100_000.0, leverage=1.0)["total_cost"]
    two = ob.leaps_plan(None, 100_000.0, leverage=2.0)["total_cost"]
    assert two > one                                 # 2x notional => more premium


def test_rows_have_contract_counts():
    pl = ob.leaps_plan(None, 100_000.0, leverage=1.0)
    for r in pl["rows"]:
        if r.get("ok"):
            assert r["contracts"] >= 1
            assert r["cost"] > 0
