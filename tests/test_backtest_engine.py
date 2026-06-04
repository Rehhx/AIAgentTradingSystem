"""Event-driven backtest engine: look-ahead firewall, accounting, fill logic, parity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from backtest import (
    HistoricDataHandler, Portfolio, SimulatedExecution, Backtest, Strategy,
)
from backtest.events import SignalEvent


def _handler(closes, symbol="AAA"):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    df = pd.DataFrame({"close": closes}, index=idx)
    return HistoricDataHandler({symbol: df})


class _Const(Strategy):
    def __init__(self, w):
        self.w = w

    def calculate_signal(self, symbol, history):
        return self.w


# --- the headline property: no look-ahead ----------------------------------

def test_get_history_never_sees_the_future():
    h = _handler([1, 2, 3, 4, 5])
    seen = 0
    while True:
        h.update_bars()
        if not h.continue_backtest:
            break
        seen += 1
        hist = h.get_history("AAA")
        assert len(hist) == seen                       # exactly the bars revealed so far
        assert hist.index[-1] == h.now()               # last visible bar IS the current one
        assert hist.iloc[-1] == seen                   # closes are 1,2,3,4,5 -> no peeking
    assert seen == 5


def test_latest_close_tracks_cursor():
    h = _handler([10, 20, 30])
    assert h.latest_close("AAA") is None               # before first bar
    h.update_bars(); assert h.latest_close("AAA") == 10
    h.update_bars(); assert h.latest_close("AAA") == 20


# --- accounting + fill logic ----------------------------------------------

def test_buy_fill_accounting_conserves_value():
    h = _handler([100, 100, 100])
    h.update_bars()
    pf = Portfolio(h, initial_capital=10_000)
    ex = SimulatedExecution(h, commission_rate=0.001, slippage_rate=0.0)
    pf.update_timeindex()
    order = pf.generate_order(SignalEvent("AAA", h.now(), 1.0))
    assert order.quantity == pytest.approx(100.0)      # 10_000 / 100
    fill = ex.execute_order(order)
    assert fill.commission == pytest.approx(100 * 100 * 0.001)
    pf.update_fill(fill)
    assert pf.positions["AAA"] == pytest.approx(100.0)
    assert pf.cash == pytest.approx(10_000 - 100 * 100 - 10)
    assert pf.current_equity() == pytest.approx(9_990)  # initial minus commission


def test_sell_reduces_position_and_adds_cash():
    h = _handler([50, 50])
    h.update_bars()
    pf = Portfolio(h, 10_000)
    ex = SimulatedExecution(h, commission_rate=0.0)
    pf.update_fill(ex.execute_order(pf.generate_order(SignalEvent("AAA", h.now(), 1.0))))
    assert pf.positions["AAA"] == pytest.approx(200.0)
    # now flatten
    order = pf.generate_order(SignalEvent("AAA", h.now(), 0.0))
    assert order.quantity == pytest.approx(-200.0)
    pf.update_fill(ex.execute_order(order))
    assert pf.positions["AAA"] == pytest.approx(0.0)
    assert pf.current_equity() == pytest.approx(10_000)


def test_unchanged_target_emits_no_order():
    h = _handler([100, 100])
    h.update_bars()
    pf = Portfolio(h, 10_000)
    pf.generate_order(SignalEvent("AAA", h.now(), 1.0))      # establishes target 1.0
    assert pf.generate_order(SignalEvent("AAA", h.now(), 1.0)) is None


# --- end-to-end behaviour --------------------------------------------------

def test_always_long_matches_buy_and_hold_zero_cost():
    closes = [100, 101, 102, 99, 105]
    h = _handler(closes)
    res = Backtest(h, _Const(1.0), Portfolio(h, 100_000),
                   SimulatedExecution(h, commission_rate=0.0)).run()
    ratio = res["equity"].iloc[-1] / 100_000
    assert ratio == pytest.approx(closes[-1] / closes[0], rel=1e-9)


def test_flat_strategy_has_no_pnl():
    h = _handler([100, 130, 80, 100])
    res = Backtest(h, _Const(0.0), Portfolio(h, 100_000),
                   SimulatedExecution(h, commission_rate=0.0)).run()
    assert res["equity"].iloc[-1] == pytest.approx(100_000)
    assert res["counts"]["fills"] == 0


def test_entry_cost_lands_on_next_bar():
    # mark-before-fill: a buy at bar 1 shows its commission as bar-2 return, == SIDE_COST
    closes = [100, 100, 100]                            # flat price isolates the cost
    h = _handler(closes)
    res = Backtest(h, _Const(1.0), Portfolio(h, 100_000),
                   SimulatedExecution(h, commission_rate=0.0003)).run()
    r = res["returns"]
    assert r.iloc[0] == pytest.approx(0.0)             # nothing held into bar 0
    assert r.iloc[1] == pytest.approx(-0.0003, abs=1e-9)  # entry cost on the next bar
