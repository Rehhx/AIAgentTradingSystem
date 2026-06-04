"""
ml — financial machine learning done correctly (BUILD_PLAN.md Tier 3)
---------------------------------------------------------------------
The point of this package is METHODOLOGY, not a magic signal. Naive ML on price
data leaks the future through two doors that standard scikit-learn workflows leave
wide open:

  1. OVERLAPPING LABELS: a triple-barrier label formed at t looks forward up to a
     horizon, so an ordinary K-fold puts a training row whose label-window overlaps
     a test row in the same split -> leakage. `cv.PurgedKFold` purges the overlap
     and embargoes the serial-correlation tail (Lopez de Prado, AFML ch. 7).
  2. SHUFFLING TIME: cross_val_score shuffles, destroying the time order. PurgedKFold
     keeps folds contiguous.

`labels.triple_barrier_labels` builds path-dependent labels; `features.make_features`
builds trailing-only (no look-ahead) features. `runners/ml_signal.py` ties it together
and judges the result with the Tier-1A deflated Sharpe — the honest test.
"""
from ml.cv import PurgedKFold
from ml.labels import get_daily_vol, triple_barrier_labels
from ml.features import make_features

__all__ = ["PurgedKFold", "get_daily_vol", "triple_barrier_labels", "make_features"]
