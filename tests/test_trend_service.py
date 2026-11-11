import json
import math

import pytest

from app.providers.variables import get_variable
from app.services.trend_analysis_service import analyze_timeseries
from tests.helpers import synthetic_monthly

T2M = get_variable("T2M")
PRECIP = get_variable("PRECTOTCORR")


def test_recovers_known_trend_and_calls_it_significant():
    r = analyze_timeseries(synthetic_monthly(slope_per_year=0.03, seed=1), T2M)
    t = r.trend
    assert t.classification == "increasing" and t.significant and t.direction == "increasing"
    assert t.sen_slope.slope_per_decade == pytest.approx(0.3, abs=0.06)
    assert t.sen_slope.ci_low_per_decade < 0.3 < t.sen_slope.ci_high_per_decade
    assert t.ols.slope_per_decade == pytest.approx(0.3, abs=0.06)
    assert t.mann_kendall.p_value < 0.05
    assert t.change.total_change == pytest.approx(0.03 * 39, abs=0.25)
    assert t.change.effect_size_label == "large"


def test_decreasing_trend():
    r = analyze_timeseries(synthetic_monthly(slope_per_year=-0.04, seed=3), T2M)
    assert r.trend.classification == "decreasing"
    assert r.trend.sen_slope.slope_per_decade < 0


def test_pure_noise_is_not_called_a_trend():
    r = analyze_timeseries(synthetic_monthly(slope_per_year=0.0, noise_sd=0.8, seed=5), T2M)
    assert r.trend.classification == "no_significant_trend"
    assert r.trend.significant is False and r.trend.direction is None
    assert "no statistically significant trend" in r.insight


def test_seasonal_cycle_does_not_create_a_trend():
    r = analyze_timeseries(synthetic_monthly(slope_per_year=0.0, amp=15, noise_sd=0.3, seed=8), T2M)
    assert r.trend.classification == "no_significant_trend"


def test_too_few_years_gives_insufficient_data_not_a_guess():
    r = analyze_timeseries(synthetic_monthly(start_year=2018, end_year=2024), T2M)
    assert r.trend.classification == "insufficient_data"
    assert r.trend.sen_slope is None and r.trend.mann_kendall is None
    assert "not enough valid data" in r.insight


def test_percent_change_is_withheld_for_celsius_but_given_for_precipitation():
    t = analyze_timeseries(synthetic_monthly(seed=1), T2M).trend.change
    assert t.pct_change is None and t.pct_change_note
    p = analyze_timeseries(
        synthetic_monthly(base=5.0, amp=2.0, slope_per_year=0.02, noise_sd=0.2, variable="PRECTOTCORR",
                          unit="mm/day", seed=1), PRECIP,
    ).trend.change
    assert p.pct_change is not None and p.pct_change > 0


def test_missing_data_is_reported_and_analysis_still_works():
    r = analyze_timeseries(synthetic_monthly(missing_frac=0.15, seed=9), T2M)
    assert r.diagnostics.missing_fraction > 0.05
    assert r.trend.classification == "increasing"
    assert "had no data" in r.insight


def test_series_arrays_are_aligned_and_chart_ready():
    r = analyze_timeseries(synthetic_monthly(seed=1), T2M)
    s = r.series
    n = len(s.monthly_dates)
    assert n == len(s.monthly_values) == len(s.monthly_anomalies) == len(s.monthly_anomalies_moving_avg_12m) == 480
    k = len(s.annual_years)
    assert k == len(s.annual_anomaly) == len(s.annual_mean) == len(s.annual_anomaly_moving_avg_5y) == len(s.trend_line) == 40
    assert s.annual_anomaly_moving_avg_5y[0] is None and s.annual_anomaly_moving_avg_5y[-1] is None
    # trend line rises by the Sen slope per year
    step = s.trend_line[1] - s.trend_line[0]
    assert step == pytest.approx(r.trend.sen_slope.slope_per_decade / 10, abs=1e-5)


def test_output_is_strict_json_with_no_nan():
    r = analyze_timeseries(synthetic_monthly(missing_frac=0.1, seed=2), T2M)
    json.dumps(r.model_dump(mode="json"), allow_nan=False)  # raises if NaN/inf sneaks in


def test_insight_numbers_come_from_the_result():
    r = analyze_timeseries(synthetic_monthly(seed=1), T2M)
    rate = f"{abs(r.trend.sen_slope.slope_per_decade):.3g}"
    assert rate in r.insight
    assert "significant upward trend" in r.insight
    assert "MERRA-2" in r.insight                       # reanalysis caveat present
    assert "25.43°N, 81.84°E" in r.insight


def test_no_causes_are_claimed():
    r = analyze_timeseries(synthetic_monthly(seed=1), T2M)
    for word in ("because of", "caused by", "climate change", "due to"):
        assert word not in r.insight.lower()


def test_methods_disagreement_is_reported_not_hidden():
    from app.algorithms.significance import classify_trend
    a = classify_trend(0.001, 10, 0.02, -0.01)
    assert a.classification == "inconclusive" and a.reasons


def test_baseline_and_autocorrelation_diagnostics_present():
    r = analyze_timeseries(synthetic_monthly(seed=1), T2M)
    assert "calendar month" in r.diagnostics.baseline
    assert r.diagnostics.lag1_autocorrelation is None or math.isfinite(r.diagnostics.lag1_autocorrelation)
    assert r.trend.mann_kendall.variance_inflation_factor >= 1.0
