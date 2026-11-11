"""Mann-Kendall trend tests.

* mann_kendall            - classic test (assumes independent observations)
* hamed_rao_mann_kendall  - variance corrected for serial autocorrelation
                            (Hamed & Rao, 1998). Use this for time series,
                            because positive autocorrelation makes the classic
                            test report too many "significant" trends.
* seasonal_mann_kendall   - Hirsch-style: tests each calendar month separately
                            across years and combines them. Seasons are treated
                            as independent, which is a simplification.

All functions refuse NaN input: missing data are removed by the caller, never
imputed here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from app.algorithms._common import clean_1d, pairwise_slopes


@dataclass(frozen=True)
class MannKendallResult:
    method: str
    n: int
    s: float
    tau: float                 # S / (number of pairs); a simple Kendall tau-a
    var_s: float               # variance actually used for z
    var_s_uncorrected: float
    variance_inflation: float  # var_s / var_s_uncorrected (1.0 = no correction)
    z: float
    p_value: float             # two-sided, uses var_s
    p_value_uncorrected: float
    trend: str                 # "increasing" | "decreasing" | "no trend" at `alpha`
    alpha: float
    n_seasons: int | None = None


def s_statistic(x: np.ndarray) -> float:
    i, j = np.triu_indices(len(x), k=1)
    return float(np.sign(x[j] - x[i]).sum())


def variance_s(x: np.ndarray) -> float:
    """Variance of S with the standard correction for tied values."""
    n = len(x)
    _, counts = np.unique(x, return_counts=True)
    t = counts[counts > 1].astype(float)
    tie_term = float((t * (t - 1) * (2 * t + 5)).sum())
    return (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0


def _z_from_s(s: float, var: float) -> float:
    if var <= 0 or s == 0:
        return 0.0
    return float((s - np.sign(s)) / np.sqrt(var))  # continuity correction


def _p_from_z(z: float) -> float:
    return float(2.0 * stats.norm.sf(abs(z)))


def _label(p: float, z: float, alpha: float) -> str:
    if p < alpha:
        return "increasing" if z > 0 else "decreasing"
    return "no trend"


def mann_kendall(x, alpha: float = 0.05) -> MannKendallResult:
    x = clean_1d(x)
    n = len(x)
    if n < 4:
        raise ValueError("Mann-Kendall needs at least 4 observations")
    s = s_statistic(x)
    var = variance_s(x)
    z = _z_from_s(s, var)
    p = _p_from_z(z)
    return MannKendallResult(
        method="mann_kendall", n=n, s=s, tau=s / (n * (n - 1) / 2), var_s=var,
        var_s_uncorrected=var, variance_inflation=1.0, z=z, p_value=p,
        p_value_uncorrected=p, trend=_label(p, z, alpha), alpha=alpha,
    )


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Biased sample autocorrelation, lags 0..nlags."""
    n = len(x)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom == 0:
        return np.zeros(nlags + 1)
    return np.array([float((xc[: n - k] * xc[k:]).sum()) / denom for k in range(nlags + 1)])


def hamed_rao_mann_kendall(x, alpha: float = 0.05) -> MannKendallResult:
    """Mann-Kendall with the Hamed-Rao (1998) autocorrelation correction.

    Steps: remove the Sen trend, rank the residuals, find lags whose rank
    autocorrelation is significant, and inflate Var(S) by n/n* computed from
    them.

    DELIBERATE DEPARTURE from the published method: the factor is floored at
    1.0, so the correction can only make the test MORE cautious than classic
    Mann-Kendall, never less. On short series (n ~ 40) the estimated factor is
    unstable: it can fall below 1 (making a trend look more significant than
    the plain test says) or even go negative (pymannkendall then returns NaN).
    For a tool whose job is to avoid false "significant" claims, the cautious
    floor is the safer choice. Both p-values are reported.
    """
    x = clean_1d(x)
    n = len(x)
    if n < 4:
        raise ValueError("Mann-Kendall needs at least 4 observations")
    s = s_statistic(x)
    var0 = variance_s(x)

    t = np.arange(1, n + 1, dtype=float)
    slope = float(np.median(pairwise_slopes(t, x)))
    ranks = stats.rankdata(x - slope * t)
    r = _acf(ranks, n - 1)
    bound = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)

    total = 0.0
    for k in range(1, n - 2):
        if abs(r[k]) > bound:
            total += (n - k) * (n - k - 1) * (n - k - 2) * r[k]
    factor = 1.0 + 2.0 * total / (n * (n - 1) * (n - 2))
    factor = max(factor, 1.0)

    var = var0 * factor
    z = _z_from_s(s, var)
    p = _p_from_z(z)
    p0 = _p_from_z(_z_from_s(s, var0))
    return MannKendallResult(
        method="mann_kendall_hamed_rao", n=n, s=s, tau=s / (n * (n - 1) / 2), var_s=var,
        var_s_uncorrected=var0, variance_inflation=float(factor), z=z, p_value=p,
        p_value_uncorrected=p0, trend=_label(p, z, alpha), alpha=alpha,
    )


def seasonal_mann_kendall(matrix, alpha: float = 0.05, min_per_season: int = 5) -> MannKendallResult:
    """Seasonal Mann-Kendall.

    `matrix` has one row per year (ascending) and one column per season
    (e.g. 12 calendar months); NaN marks a missing value. Seasons with fewer
    than `min_per_season` valid values are skipped.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D (years x seasons)")
    s_tot = var_tot = pairs = 0.0
    used = n_obs = 0
    for k in range(m.shape[1]):
        col = m[:, k]
        col = col[~np.isnan(col)]
        if len(col) < min_per_season:
            continue
        s_tot += s_statistic(col)
        var_tot += variance_s(col)
        pairs += len(col) * (len(col) - 1) / 2
        n_obs += len(col)
        used += 1
    if used == 0:
        raise ValueError("no season has enough valid values for the seasonal test")
    z = _z_from_s(s_tot, var_tot)
    p = _p_from_z(z)
    return MannKendallResult(
        method="seasonal_mann_kendall", n=n_obs, s=s_tot, tau=s_tot / pairs if pairs else 0.0,
        var_s=var_tot, var_s_uncorrected=var_tot, variance_inflation=1.0, z=z, p_value=p,
        p_value_uncorrected=p, trend=_label(p, z, alpha), alpha=alpha, n_seasons=used,
    )
