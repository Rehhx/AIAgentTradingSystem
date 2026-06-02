"""Black-Scholes pricing/delta used by the LEAPS leverage analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pytest
from runners.options_leverage import bs_call, bs_delta, _ncdf


def test_ncdf_symmetry():
    assert _ncdf(0) == pytest.approx(0.5, abs=1e-9)
    assert _ncdf(-1) + _ncdf(1) == pytest.approx(1.0, abs=1e-9)


def test_call_at_expiry_is_intrinsic():
    assert bs_call(120, 100, 0, 0.2, 0.03) == pytest.approx(20.0)      # ITM
    assert bs_call(80, 100, 0, 0.2, 0.03) == pytest.approx(0.0)        # OTM -> 0


def test_call_never_below_intrinsic():
    for S in (80, 100, 120, 150):
        c = bs_call(S, 100, 1.0, 0.25, 0.03)
        assert c >= max(0.0, S - 100) - 1e-9


def test_call_monotonic_in_spot():
    prev = -1
    for S in range(60, 160, 10):
        c = bs_call(S, 100, 1.0, 0.2, 0.03)
        assert c > prev
        prev = c


def test_deep_itm_delta_high_atm_mid():
    # 10% ITM 1yr call ~ 0.8 delta (the LEAPS share-replacement target)
    d_itm = bs_delta(100, 90, 1.0, 0.2, 0.03)
    assert 0.70 < d_itm < 0.92
    d_atm = bs_delta(100, 100, 1.0, 0.2, 0.03)
    assert 0.45 < d_atm < 0.70
    assert d_itm > d_atm                       # deeper ITM => higher delta


def test_delta_bounded_0_1():
    for S in (50, 100, 200):
        d = bs_delta(S, 100, 1.0, 0.3, 0.03)
        assert 0.0 <= d <= 1.0


def test_higher_vol_higher_call():
    lo = bs_call(100, 100, 1.0, 0.10, 0.03)
    hi = bs_call(100, 100, 1.0, 0.40, 0.03)
    assert hi > lo                             # more vol => more option value
