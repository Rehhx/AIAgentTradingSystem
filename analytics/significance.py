"""
analytics/significance.py
-------------------------
Deflated Sharpe Ratio and friends (Bailey & Lopez de Prado, 2012-2014).

The Sharpe ratio you read off a backtest is INFLATED by two things this module
corrects for:
  1. Non-normal returns (fat tails, negative skew) make the naive Sharpe's
     standard error wrong -> the Probabilistic Sharpe Ratio (PSR) fixes the s.e.
  2. SELECTION BIAS: if you try N strategies and report the best, the maximum
     Sharpe is upward-biased even if every strategy is truly worthless. The
     Deflated Sharpe Ratio (DSR) deflates by the expected maximum Sharpe under
     the null of N independent worthless trials.

CONVENTION: SR here is the NON-annualized (per-observation) Sharpe — mean/std of
the raw return series. Annualization is a separate, cosmetic sqrt(periods)
scaling; the PSR/DSR statistics are defined on the per-period SR and the
observation count n. Mixing the two is the most common implementation bug, so
every function below is explicit about which it expects.

References:
  Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier", J. Risk.
  Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", J. Portfolio Mgmt.
"""
from __future__ import annotations

import math

import numpy as np

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# normal CDF / inverse CDF  (no scipy dependency)
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """inverse standard-normal CDF (quantile) via Acklam's rational
    approximation; |abs error| < 1.2e-9 for p in (0, 1)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------------------
# Sharpe moments
# ---------------------------------------------------------------------------

def sharpe_stats(returns) -> dict:
    """per-observation Sharpe + the higher moments PSR/DSR need.

    Returns sr (per-period), n (obs), skew, kurt (non-excess: normal == 3)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3:
        return {"sr": 0.0, "n": n, "skew": 0.0, "kurt": 3.0}
    sd = r.std(ddof=1)
    if sd == 0:
        return {"sr": 0.0, "n": n, "skew": 0.0, "kurt": 3.0}
    mu = r.mean()
    z = (r - mu) / sd
    return {"sr": float(mu / sd), "n": int(n),
            "skew": float(np.mean(z ** 3)), "kurt": float(np.mean(z ** 4))}


def annualized_sharpe(returns, periods: int = 252) -> float:
    """convenience: per-period SR scaled by sqrt(periods)."""
    return sharpe_stats(returns)["sr"] * math.sqrt(periods)


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(sr: float, n: int, skew: float, kurt: float,
                               sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark) given the observed per-period SR `sr` over `n`
    observations and the return distribution's skew/kurtosis. All SR per-period.

    Reduces to the textbook normal case when skew=0, kurt=3."""
    if n < 2:
        return 0.5
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    denom = math.sqrt(max(1e-12, var_term))
    stat = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return norm_cdf(stat)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] across `n_trials` INDEPENDENT strategies each with true
    per-period SR = 0, where the cross-trial SR estimates have variance
    `sr_variance`. This is the deflation hurdle SR* in the DSR.

    Closed form (expected max of N gaussian draws), Bailey & Lopez de Prado 2014.
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    v = math.sqrt(sr_variance)
    return v * ((1 - EULER_GAMMA) * norm_ppf(1 - 1.0 / n_trials)
                + EULER_GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe_ratio(sr: float, n: int, skew: float, kurt: float,
                          n_trials: int, sr_variance: float) -> dict:
    """Deflated Sharpe Ratio = PSR evaluated at the selection hurdle SR*.

    Returns the DSR, the hurdle SR*, and PSR-vs-zero (the un-deflated baseline)
    so the deflation is auditable: DSR <= psr_vs_zero always."""
    sr_star = expected_max_sharpe(n_trials, sr_variance)
    return {
        "dsr": probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr_star),
        "sr_star": sr_star,
        "psr_vs_zero": probabilistic_sharpe_ratio(sr, n, skew, kurt, 0.0),
    }


def min_track_record_length(sr: float, skew: float, kurt: float,
                            sr_benchmark: float = 0.0, prob: float = 0.95) -> float:
    """smallest number of observations for the PSR to exceed `prob` confidence
    that true SR > sr_benchmark. inf if the observed SR doesn't beat the hurdle.
    All SR per-period."""
    if sr <= sr_benchmark:
        return math.inf
    z = norm_ppf(prob)
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return 1.0 + var_term * (z / (sr - sr_benchmark)) ** 2


# ---------------------------------------------------------------------------
# convenience: deflate the winner by the spread of all trials' Sharpes
# ---------------------------------------------------------------------------

def dsr_from_trials(best_returns, trial_sharpes, periods: int = 252) -> dict:
    """Deflate the best strategy's Sharpe by the variance of per-period Sharpes
    across ALL strategies tried (including the winner).

    best_returns   : the winning strategy's raw return series
    trial_sharpes  : per-period SRs of every strategy tried
    Returns a report dict with both per-period and annualized figures.
    """
    s = sharpe_stats(best_returns)
    ts = np.asarray(trial_sharpes, dtype=float)
    ts = ts[~np.isnan(ts)]
    n_trials = len(ts)
    sr_var = float(np.var(ts, ddof=1)) if n_trials > 1 else 0.0
    d = deflated_sharpe_ratio(s["sr"], s["n"], s["skew"], s["kurt"], n_trials, sr_var)
    mtrl = min_track_record_length(s["sr"], s["skew"], s["kurt"],
                                   sr_benchmark=d["sr_star"], prob=0.95)
    rt = math.sqrt(periods)
    return {
        "sr_annual": s["sr"] * rt, "sr_period": s["sr"], "n_obs": s["n"],
        "skew": s["skew"], "kurt": s["kurt"],
        "n_trials": n_trials, "sr_variance": sr_var,
        "sr_star_period": d["sr_star"], "sr_star_annual": d["sr_star"] * rt,
        "psr_vs_zero": d["psr_vs_zero"], "dsr": d["dsr"],
        "min_track_record_length": mtrl,
    }
