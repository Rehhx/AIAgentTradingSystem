"""Order construction in ExecutionAgent: symbol mapping (BRK-B->BRK.B, crypto),
extended-hours rules, and the reverse mapping in get_positions. Fake client, no network."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from agents.execution_agent import ExecutionAgent


class _Order:
    id = "fake-id"
    client_order_id = "coid"
    status = "accepted"


class _FakeClient:
    def __init__(self):
        self.last_req = None

    def submit_order(self, req):
        self.last_req = req
        return _Order()


def _agent():
    a = ExecutionAgent(api_key="", api_secret="")   # starts simulated
    a.simulated = False                              # force the live path
    a.client = _FakeClient()
    return a


def test_class_share_dash_to_dot():
    a = _agent()
    a._submit_alpaca("BRK-B", "buy", 1, 0, "market", "day", None)
    assert a.client.last_req.symbol == "BRK.B"


def test_crypto_symbol_and_gtc():
    a = _agent()
    from alpaca.trading.enums import TimeInForce
    a._submit_alpaca("BTC-USD", "buy", 0, 500, "market", "day", None)
    req = a.client.last_req
    assert req.symbol == "BTC/USD"
    assert req.time_in_force == TimeInForce.GTC      # crypto must be GTC (24/7)


def test_extended_hours_requires_limit():
    a = _agent()
    with pytest.raises(ValueError):
        a._submit_alpaca("AAPL", "buy", 5, 0, "market", "day", None, extended_hours=True)


def test_extended_hours_limit_sets_flag_and_day():
    a = _agent()
    from alpaca.trading.enums import TimeInForce
    a._submit_alpaca("AAPL", "buy", 5, 0, "limit", "day", 312.0, extended_hours=True)
    req = a.client.last_req
    assert req.extended_hours is True
    assert req.time_in_force == TimeInForce.DAY
    assert float(req.limit_price) == 312.0


def test_plain_equity_unchanged():
    a = _agent()
    a._submit_alpaca("AAPL", "sell", 3, 0, "market", "day", None)
    assert a.client.last_req.symbol == "AAPL"


def test_get_positions_reverse_maps_dot_to_dash():
    class _Pos:
        symbol = "BRK.B"; qty = "2"; avg_entry_price = "100"
        unrealized_pl = "5"; market_value = "210"

    class _C:
        def get_all_positions(self):
            return [_Pos()]
    a = _agent()
    a.client = _C()
    pos = a.get_positions()
    assert pos[0]["symbol"] == "BRK-B"               # mapped back to yfinance convention
