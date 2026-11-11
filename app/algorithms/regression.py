"""Linear (OLS) trend with autocorrelation-robust standard errors.

Plain OLS standard errors assume independent residuals. Climate series are
usually serially correlated, which makes plain OLS confidence intervals too
narrow. We therefore report Newey-West (HAC, Bartlett kernel) errors with a
small-sample correction as the primary uncertainty, and keep the naive p-value
only for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from app.algorithms._common import clean_1d


@dataclass(frozen=True)
class OlsResult:
    slope: float
    intercept: float
    stderr_hac: float
    p_value_hac: float
    ci_low: float
    ci_high: float
    stderr_naive: float
    p_value_naive: float
    r_squared: float
    n: int
    hac_lags: int
    confidence_level: float


def default_hac_lags(n: int) -> int:
    """Newey-West rule of thumb: floor(4 * (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def _p_from_t(slope: float, se: float, df: int) -> float:
    if se == 0:
        return 1.0 if slope == 0 else 0.0
    return float(2.0 * stats.t.sf(abs(slope / se), df))


def ols_trend(x, y, alpha: float = 0.05, hac_lags: int | None = None) -> OlsResult:
    x = clean_1d(x, "x")
    y = clean_1d(y, "y")
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n < 4:
        raise ValueError("OLS trend needs at least 4 points")

    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    df = n - 2

    sigma2 = float(resid @ resid) / df
    se_naive = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

    L = default_hac_lags(n) if hac_lags is None else hac_lags
    xu = X * resid[:, None]
    S = xu.T @ xu
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        G = xu[lag:].T @ xu[:-lag]
        S += w * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv * (n / df)
    se_hac = float(np.sqrt(max(cov[1, 1], 0.0)))

    slope = float(beta[1])
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 0.0 if sst == 0 else 1.0 - float(resid @ resid) / sst

    return OlsResult(
        slope=slope, intercept=float(beta[0]),
        stderr_hac=se_hac, p_value_hac=_p_from_t(slope, se_hac, df),
        ci_low=slope - tcrit * se_hac, ci_high=slope + tcrit * se_hac,
        stderr_naive=se_naive, p_value_naive=_p_from_t(slope, se_naive, df),
        r_squared=r2, n=n, hac_lags=L, confidence_level=1 - alpha,
    )
