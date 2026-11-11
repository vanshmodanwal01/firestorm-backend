"""Sen's slope (Theil-Sen) estimator with a confidence interval.

The slope is the median of all pairwise slopes, so a few extreme years barely
move it. The confidence interval follows Sen (1968) and is driven by Var(S);
pass the autocorrelation-corrected variance from the Hamed-Rao test so the
interval widens when the series is serially correlated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from app.algorithms._common import clean_1d, pairwise_slopes
from app.algorithms.mann_kendall import variance_s


@dataclass(frozen=True)
class SenSlopeResult:
    slope: float            # units of y per unit of x
    intercept: float        # median(y) - slope * median(x)
    ci_low: float
    ci_high: float
    confidence_level: float
    n_slopes: int


def sens_slope(x, y, var_s: float | None = None, alpha: float = 0.05) -> SenSlopeResult:
    x = clean_1d(x, "x")
    y = clean_1d(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        raise ValueError("Sen's slope needs at least 3 points")

    slopes = np.sort(pairwise_slopes(x, y))
    n_slopes = len(slopes)
    if n_slopes == 0:
        raise ValueError("all x values are identical; slope is undefined")

    med = float(np.median(slopes))
    intercept = float(np.median(y) - med * np.median(x))

    if var_s is None:
        var_s = variance_s(y)
    c = float(stats.norm.ppf(1 - alpha / 2) * np.sqrt(max(var_s, 0.0)))
    hi = min(int(np.round((n_slopes + c) / 2.0)), n_slopes - 1)
    lo = max(int(np.round((n_slopes - c) / 2.0)) - 1, 0)
    return SenSlopeResult(med, intercept, float(slopes[lo]), float(slopes[hi]), 1 - alpha, n_slopes)


def seasonal_sens_slope(years, matrix, min_per_season: int = 5) -> float | None:
    """Median of within-season pairwise slopes (units of value per year)."""
    years = np.asarray(years, dtype=float)
    m = np.asarray(matrix, dtype=float)
    collected: list[np.ndarray] = []
    for k in range(m.shape[1]):
        col = m[:, k]
        ok = ~np.isnan(col)
        if ok.sum() < min_per_season:
            continue
        collected.append(pairwise_slopes(years[ok], col[ok]))
    if not collected:
        return None
    allslopes = np.concatenate(collected)
    return float(np.median(allslopes)) if len(allslopes) else None
