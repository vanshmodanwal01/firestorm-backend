"""Turning test results into a classification, and separating statistical
significance from practical importance.

Rules (deliberately conservative):
* A trend is called increasing/decreasing ONLY if the autocorrelation-corrected
  Mann-Kendall p-value is below alpha AND Mann-Kendall, Sen's slope and OLS all
  agree on the direction.
* If the test is significant but the methods disagree on direction, the result
  is "inconclusive".
* Otherwise it is "no_significant_trend". That is absence of evidence, not
  proof that nothing changed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_YEARS_FOR_TREND = 10


@dataclass(frozen=True)
class Assessment:
    classification: str
    direction: str | None
    significant: bool
    reasons: list[str]


def _sign(v: float) -> int:
    return int(np.sign(v))


def classify_trend(mk_p: float, mk_s: float, sen_slope: float, ols_slope: float,
                   alpha: float = 0.05) -> Assessment:
    if mk_p >= alpha:
        return Assessment(
            "no_significant_trend", None, False,
            [f"Mann-Kendall p-value {mk_p:.3g} is not below alpha {alpha}."],
        )
    signs = {_sign(mk_s), _sign(sen_slope), _sign(ols_slope)}
    if len(signs) != 1 or 0 in signs:
        return Assessment(
            "inconclusive", None, False,
            ["Mann-Kendall is significant, but Mann-Kendall, Sen's slope and OLS do not agree on direction."],
        )
    direction = "increasing" if _sign(mk_s) > 0 else "decreasing"
    return Assessment(direction, direction, True, [])


def effect_size(total_change: float, residual_sd: float) -> tuple[float | None, str | None]:
    """Size of the change relative to normal year-to-year variation.

    effect_size = |total change over the period| / standard deviation of annual
    values around the trend line. Labels are a simple heuristic for explaining
    results to students (<1 small, 1-2 moderate, >=2 large), not a scientific
    standard; whether a change matters environmentally depends on the variable.
    """
    if not np.isfinite(residual_sd) or residual_sd <= 0:
        return None, None
    es = abs(total_change) / residual_sd
    label = "small" if es < 1 else "moderate" if es < 2 else "large"
    return float(es), label
