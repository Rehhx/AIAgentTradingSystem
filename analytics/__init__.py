"""
analytics — statistical-rigor toolkit (BUILD_PLAN.md Tier 1A)
-------------------------------------------------------------
Quantifies whether a backtested edge survives the multiple-testing problem — the
#1 failure mode in quant research. Three independent tools:

  significance   Deflated Sharpe Ratio, Probabilistic Sharpe, min track record
  pbo            Probability of Backtest Overfitting (CSCV)
  reality_check  White's Reality Check + Hansen's SPA (data-snooping p-values)

Pure numpy/math — no scipy dependency, so every function is self-contained and
unit-tested.
"""
from analytics.significance import (
    norm_cdf, norm_ppf, sharpe_stats, annualized_sharpe,
    probabilistic_sharpe_ratio, expected_max_sharpe, deflated_sharpe_ratio,
    min_track_record_length, dsr_from_trials,
)
from analytics.pbo import cscv_pbo
from analytics.reality_check import whites_reality_check, hansen_spa

__all__ = [
    "norm_cdf", "norm_ppf", "sharpe_stats", "annualized_sharpe",
    "probabilistic_sharpe_ratio", "expected_max_sharpe", "deflated_sharpe_ratio",
    "min_track_record_length", "dsr_from_trials",
    "cscv_pbo", "whites_reality_check", "hansen_spa",
]
