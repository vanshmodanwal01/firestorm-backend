"""Cross-checks of our trend algorithms against independent reference code.

pymannkendall, scipy.stats.theilslopes and statsmodels are only used here, to
confirm that our own implementations produce the same numbers.
"""
import numpy as np
import pytest
from scipy import stats

from app.algorithms.mann_kendall import (
    hamed_rao_mann_kendall,
    mann_kendall,
    seasonal_mann_kendall,
)
from app.algorithms.regression import default_hac_lags, ols_trend
from app.algorithms.sen_slope import sens_slope, seasonal_sens_slope
from app.algorithms.significance import classify_trend, effect_size


def ar1(n, phi, rng, sd=1.0):
    e = rng.normal(0, sd, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


# ------------------------------------------------------------ Mann-Kendall
def test_mann_kendall_matches_pymannkendall():
    pmk = pytest.importorskip("pymannkendall")
    rng = np.random.default_rng(3)
    for _ in range(5):
        x = np.cumsum(rng.normal(0.1, 1, 40))
        ours = mann_kendall(x)
        ref = pmk.original_test(x)
        assert ours.s == ref.s
        assert ours.z == pytest.approx(ref.z, abs=1e-9)
        assert ours.p_value == pytest.approx(ref.p, abs=1e-9)
        assert ours.var_s == pytest.approx(ref.var_s, abs=1e-9)


def test_mann_kendall_handles_ties_like_reference():
    pmk = pytest.importorskip("pymannkendall")
    x = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10], dtype=float)
    ours, ref = mann_kendall(x), pmk.original_test(x)
    assert ours.var_s == pytest.approx(ref.var_s)
    assert ours.p_value == pytest.approx(ref.p)


def test_hamed_rao_matches_pymannkendall():
    pmk = pytest.importorskip("pymannkendall")
    rng = np.random.default_rng(11)
    matched = floored = 0
    for _ in range(10):
        x = ar1(40, 0.5, rng) + 0.03 * np.arange(40)
        ours = hamed_rao_mann_kendall(x)
        ref = pmk.hamed_rao_modification_test(x)
        ref_factor = ref.var_s / ours.var_s_uncorrected
        if ref_factor >= 1:
            # identical to the published method whenever it widens uncertainty
            assert ours.var_s == pytest.approx(ref.var_s, rel=1e-6)
            assert ours.p_value == pytest.approx(ref.p, abs=1e-6)
            matched += 1
        else:
            # deliberate departure: never narrower than classic Mann-Kendall
            assert ours.variance_inflation == 1.0
            assert ours.var_s == pytest.approx(ours.var_s_uncorrected)
            floored += 1
    assert matched >= 2 and floored >= 1  # both branches were exercised


def test_correction_never_makes_test_less_cautious_than_classic():
    rng = np.random.default_rng(77)
    for _ in range(100):
        x = ar1(40, 0.4, rng) + 0.02 * np.arange(40)
        r = hamed_rao_mann_kendall(x)
        assert r.variance_inflation >= 1.0
        assert r.p_value >= r.p_value_uncorrected - 1e-12


def test_autocorrelation_correction_reduces_false_positives():
    """With no real trend, plain MK on autocorrelated data cries wolf too often."""
    rng = np.random.default_rng(42)
    plain = corrected = 0
    sims = 400
    for _ in range(sims):
        x = ar1(40, 0.6, rng)
        plain += mann_kendall(x).p_value < 0.05
        corrected += hamed_rao_mann_kendall(x).p_value < 0.05
    assert plain / sims > 0.10          # clearly inflated above the nominal 5%
    assert corrected < plain            # correction helps
    assert corrected / sims < plain / sims - 0.03


def test_strong_trend_detected_and_noise_not():
    rng = np.random.default_rng(5)
    t = np.arange(40)
    assert hamed_rao_mann_kendall(0.1 * t + rng.normal(0, 0.3, 40)).trend == "increasing"
    assert hamed_rao_mann_kendall(-0.1 * t + rng.normal(0, 0.3, 40)).trend == "decreasing"
    assert hamed_rao_mann_kendall(rng.normal(0, 1, 40)).p_value > 0.05


def test_constant_series_gives_valid_numbers_not_nan():
    r = hamed_rao_mann_kendall(np.ones(20))
    assert r.p_value == 1.0 and r.s == 0 and np.isfinite(r.z)
    o = ols_trend(np.arange(20.0), np.ones(20))
    assert np.isfinite(o.p_value_hac) and o.p_value_hac > 0.9  # no trend, no NaN


def test_nan_input_is_refused_not_silently_dropped():
    with pytest.raises(ValueError):
        mann_kendall([1, 2, np.nan, 4, 5])


def test_seasonal_mann_kendall_detects_trend_in_every_season():
    rng = np.random.default_rng(8)
    years = np.arange(30)
    seasonal_cycle = 5 * np.sin(2 * np.pi * np.arange(12) / 12)
    m = np.array([[seasonal_cycle[k] + 0.05 * y + rng.normal(0, 0.3) for k in range(12)] for y in years])
    r = seasonal_mann_kendall(m)
    assert r.trend == "increasing" and r.n_seasons == 12
    slope = seasonal_sens_slope(years + 1990, m)
    assert slope == pytest.approx(0.05, abs=0.01)


def test_seasonal_skips_sparse_seasons():
    m = np.full((30, 3), np.nan)
    m[:, 0] = np.arange(30)
    m[:3, 1] = [1, 2, 3]  # too sparse
    r = seasonal_mann_kendall(m)
    assert r.n_seasons == 1


# --------------------------------------------------------------- Sen slope
def test_sen_exact_on_perfect_line():
    x = np.arange(1990, 2020, dtype=float)
    r = sens_slope(x, 2.0 * x + 1.0)
    assert r.slope == pytest.approx(2.0) and r.intercept == pytest.approx(1.0)


def test_sen_matches_scipy_theilslopes_including_interval():
    rng = np.random.default_rng(21)
    x = np.arange(1985, 2025, dtype=float)
    y = 0.03 * (x - 1985) + rng.normal(0, 0.2, len(x))
    ours = sens_slope(x, y)  # uncorrected variance
    med, intercept, lo, hi = stats.theilslopes(y, x, alpha=0.95)
    assert ours.slope == pytest.approx(med)
    assert ours.intercept == pytest.approx(intercept)
    assert ours.ci_low == pytest.approx(lo)
    assert ours.ci_high == pytest.approx(hi)


def test_sen_is_robust_to_one_wild_outlier():
    x = np.arange(30, dtype=float)
    y = 0.5 * x
    y[-1] = 500  # an extreme value at the end has high leverage on OLS
    assert sens_slope(x, y).slope == pytest.approx(0.5, abs=0.05)
    assert ols_trend(x, y).slope > 3.0  # OLS is dragged away, Sen is not


def test_sen_interval_widens_with_autocorrelation_correction():
    rng = np.random.default_rng(2)
    x = np.arange(40, dtype=float)
    y = 0.05 * x + ar1(40, 0.6, rng)
    mk = hamed_rao_mann_kendall(y)
    corrected = sens_slope(x, y, var_s=mk.var_s)
    naive = sens_slope(x, y)
    if mk.variance_inflation > 1:
        assert (corrected.ci_high - corrected.ci_low) >= (naive.ci_high - naive.ci_low)


# --------------------------------------------------------------------- OLS
def test_ols_exact_on_perfect_line():
    x = np.arange(20, dtype=float)
    r = ols_trend(x, 3 * x + 2)
    assert r.slope == pytest.approx(3) and r.intercept == pytest.approx(2)
    assert r.r_squared == pytest.approx(1.0)


def test_ols_matches_scipy_linregress_for_naive_numbers():
    rng = np.random.default_rng(9)
    x = np.arange(40, dtype=float)
    y = 0.1 * x + rng.normal(0, 1, 40)
    ours = ols_trend(x, y)
    ref = stats.linregress(x, y)
    assert ours.slope == pytest.approx(ref.slope)
    assert ours.stderr_naive == pytest.approx(ref.stderr)
    assert ours.p_value_naive == pytest.approx(ref.pvalue)
    assert ours.r_squared == pytest.approx(ref.rvalue ** 2)


def test_ols_hac_stderr_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(13)
    x = np.arange(40, dtype=float)
    y = 0.05 * x + ar1(40, 0.5, rng)
    ours = ols_trend(x, y)
    L = default_hac_lags(40)
    fit = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True}
    )
    assert ours.hac_lags == L
    assert ours.stderr_hac == pytest.approx(fit.bse[1], rel=1e-6)


# ------------------------------------------------------------ classification
def test_classification_requires_agreement_and_significance():
    assert classify_trend(0.001, 50, 0.03, 0.03).classification == "increasing"
    assert classify_trend(0.001, -50, -0.03, -0.03).classification == "decreasing"
    assert classify_trend(0.20, 50, 0.03, 0.03).classification == "no_significant_trend"
    assert classify_trend(0.001, 50, 0.03, -0.01).classification == "inconclusive"
    assert classify_trend(0.001, 50, 0.0, 0.03).classification == "inconclusive"


def test_significant_flag_false_unless_clear_direction():
    assert classify_trend(0.001, 50, 0.03, 0.03).significant is True
    assert classify_trend(0.001, 50, 0.03, -0.03).significant is False


def test_effect_size_labels():
    assert effect_size(0.5, 1.0)[1] == "small"
    assert effect_size(1.5, 1.0)[1] == "moderate"
    assert effect_size(-3.0, 1.0)[1] == "large"
    assert effect_size(1.0, 0.0) == (None, None)
