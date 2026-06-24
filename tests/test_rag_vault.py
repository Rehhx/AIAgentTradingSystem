"""RAG Vault sentiment overlay: tilt math, gating, and the offline fail-safe."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agents.rag_vault import apply_sentiment_overlay


class _StubClient:
    """Stands in for the live vault; returns canned verdicts (or raises)."""
    def __init__(self, verdicts=None, raises=None):
        self._verdicts = verdicts or {}
        self._raises = raises

    def signals(self, tickers, horizon=None):
        if self._raises:
            raise self._raises
        return [self._verdicts[t] for t in tickers if t in self._verdicts]


def _v(ticker, direction, strength=0.5, confidence="high", coverage=True):
    return {"ticker": ticker, "direction": direction, "strength": strength,
            "confidence": confidence, "coverage": coverage, "as_of": "2026-06-22"}


def test_long_verdict_boosts_within_cap():
    w = {"NVDA": 0.10}
    client = _StubClient({"NVDA": _v("NVDA", "long", strength=1.0)})
    out = apply_sentiment_overlay(w, tilt=0.25, client=client, verbose=False)
    assert out["NVDA"] == pytest.approx(0.10 * 1.25)        # full strength -> +tilt


def test_short_verdict_trims_toward_flat():
    w = {"AMD": 0.10}
    client = _StubClient({"AMD": _v("AMD", "short", strength=1.0)})
    out = apply_sentiment_overlay(w, tilt=0.25, client=client, verbose=False)
    assert out["AMD"] == pytest.approx(0.10 * 0.75)         # trimmed, never negative
    assert out["AMD"] >= 0.0


def test_strength_scales_the_tilt():
    w = {"NVDA": 0.10}
    client = _StubClient({"NVDA": _v("NVDA", "long", strength=0.4)})
    out = apply_sentiment_overlay(w, tilt=0.25, client=client, verbose=False)
    assert out["NVDA"] == pytest.approx(0.10 * (1 + 0.25 * 0.4))


def test_flat_and_uncovered_are_untouched():
    w = {"AAA": 0.10, "BBB": 0.10}
    client = _StubClient({"AAA": _v("AAA", "flat"),
                          "BBB": _v("BBB", "long", coverage=False)})
    out = apply_sentiment_overlay(w, tilt=0.25, client=client, verbose=False)
    assert out == w


def test_low_confidence_is_gated_out():
    w = {"NVDA": 0.10}
    client = _StubClient({"NVDA": _v("NVDA", "long", confidence="low")})
    out = apply_sentiment_overlay(w, tilt=0.25, client=client,
                                  min_confidence="medium", verbose=False)
    assert out["NVDA"] == pytest.approx(0.10)               # below gate -> unchanged


def test_cash_park_tickers_are_skipped():
    w = {"BIL": 0.30, "NVDA": 0.10}
    client = _StubClient({"NVDA": _v("NVDA", "long", strength=1.0)})
    out = apply_sentiment_overlay(w, tilt=0.25, client=client, verbose=False)
    assert out["BIL"] == pytest.approx(0.30)                # never queried/tilted
    assert out["NVDA"] == pytest.approx(0.125)


def test_offline_vault_returns_book_unchanged():
    """The headline guarantee: an unreachable vault must not break the live path."""
    w = {"NVDA": 0.10, "AMD": 0.20}
    client = _StubClient(raises=ConnectionError("refused"))
    out = apply_sentiment_overlay(w, tilt=0.25, client=client, verbose=False)
    assert out == w


def test_empty_or_zero_tilt_is_noop():
    assert apply_sentiment_overlay({}, client=_StubClient(), verbose=False) == {}
    w = {"NVDA": 0.10}
    client = _StubClient({"NVDA": _v("NVDA", "long", strength=1.0)})
    assert apply_sentiment_overlay(w, tilt=0.0, client=client, verbose=False) == w


# --- gate mode: trade longs / drop shorts / concentrate into matches ----------

def test_gate_drops_short_and_reallocates_to_long():
    w = {"NVDA": 0.10, "HPQ": 0.10}
    client = _StubClient({"NVDA": _v("NVDA", "long"), "HPQ": _v("HPQ", "short")})
    out = apply_sentiment_overlay(w, mode="gate", client=client, verbose=False)
    assert out["HPQ"] == 0.0                                # short -> dropped
    assert out["NVDA"] == pytest.approx(0.20)              # gets the freed 0.10
    assert sum(out.values()) == pytest.approx(sum(w.values()))   # invested preserved


def test_gate_reallocates_proportional_to_weight():
    w = {"AAA": 0.20, "BBB": 0.10, "SHT": 0.30}
    client = _StubClient({"AAA": _v("AAA", "long"), "BBB": _v("BBB", "long"),
                          "SHT": _v("SHT", "short")})
    out = apply_sentiment_overlay(w, mode="gate", client=client, verbose=False)
    assert out["SHT"] == 0.0
    assert out["AAA"] == pytest.approx(0.20 + 0.30 * (0.20 / 0.30))   # 0.40
    assert out["BBB"] == pytest.approx(0.10 + 0.30 * (0.10 / 0.30))   # 0.20


def test_gate_flat_and_uncovered_left_to_algorithm():
    w = {"AAA": 0.10, "BBB": 0.10}
    client = _StubClient({"AAA": _v("AAA", "flat"),
                          "BBB": _v("BBB", "long", coverage=False)})
    out = apply_sentiment_overlay(w, mode="gate", client=client, verbose=False)
    assert out == w                                         # nothing dropped, nothing freed


def test_gate_short_with_no_long_target_frees_to_cash():
    w = {"HPQ": 0.10, "CCC": 0.10}
    client = _StubClient({"HPQ": _v("HPQ", "short"), "CCC": _v("CCC", "flat")})
    out = apply_sentiment_overlay(w, mode="gate", client=client, verbose=False)
    assert out["HPQ"] == 0.0                                # dropped, freed stays out
    assert out["CCC"] == pytest.approx(0.10)               # flat untouched
    assert sum(out.values()) == pytest.approx(0.10)        # 10% now idle -> cash-parked downstream


def test_gate_offline_vault_returns_book_unchanged():
    w = {"NVDA": 0.10, "HPQ": 0.20}
    client = _StubClient(raises=ConnectionError("refused"))
    out = apply_sentiment_overlay(w, mode="gate", client=client, verbose=False)
    assert out == w
