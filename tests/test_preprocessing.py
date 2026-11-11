from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.services.preprocessing_service import (
    annual_series, build_monthly_series, flag_outliers, monthly_anomalies,
)
from tests.helpers import make_timeseries, synthetic_monthly


def test_missing_values_stay_missing_and_are_never_filled():
    ts = synthetic_monthly(missing_frac=0.1, seed=4)
    md = build_monthly_series(ts)
    assert md.values.isna().sum() > 0
    # each original value is unchanged, and each missing one is still missing
    original = {d: v for d, v in zip(ts.dates, ts.values)}
    for stamp, v in md.values.items():
        o = original.get(stamp.date())
        if o is None:
            assert np.isnan(v)
        else:
            assert v == o


def test_duplicates_are_reported_not_silently_dropped():
    dates = [date(2020, 1, 1), date(2020, 2, 1), date(2020, 2, 1), date(2020, 3, 1)]
    md = build_monthly_series(make_timeseries(dates, [1.0, 2.0, 99.0, 3.0]))
    assert any("duplicate" in n for n in md.notes)
    assert md.values[pd.Timestamp("2020-02-01")] == 2.0  # first occurrence kept


def test_gaps_in_dates_are_inserted_as_missing_and_reported():
    dates = [date(2020, 1, 1), date(2020, 2, 1), date(2020, 5, 1)]
    md = build_monthly_series(make_timeseries(dates, [1.0, 2.0, 5.0]))
    assert len(md.values) == 5
    assert md.values.isna().sum() == 2
    assert any("inserted as missing" in n for n in md.notes)


def test_leading_and_trailing_empty_months_are_trimmed_with_a_note():
    dates = [date(2020, m, 1) for m in range(1, 9)]
    vals = [None, None, 3.0, 4.0, 5.0, 6.0, None, None]
    md = build_monthly_series(make_timeseries(dates, vals))
    assert len(md.values) == 4
    assert any("leading" in n for n in md.notes) and any("trailing" in n for n in md.notes)


def test_all_missing_gives_empty_series_and_message():
    md = build_monthly_series(make_timeseries([date(2020, 1, 1), date(2020, 2, 1)], [None, None]))
    assert md.values.empty and any("No valid observations" in n for n in md.notes)


def test_outliers_are_flagged_but_kept():
    ts = synthetic_monthly(noise_sd=0.3, seed=2)
    ts.values[100] = ts.values[100] + 40.0
    md = build_monthly_series(ts)
    stamp = pd.Timestamp(ts.dates[100])
    assert stamp in md.outlier_dates
    assert md.values[stamp] == ts.values[100]          # still there, unchanged
    assert any("KEPT" in n for n in md.notes)


def test_no_outliers_in_clean_data():
    anomalies = monthly_anomalies(build_monthly_series(synthetic_monthly(noise_sd=0.3)).values)
    assert flag_outliers(anomalies) == []


def test_daily_is_averaged_to_monthly_with_minimum_valid_days():
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(91)]      # Jan, Feb, Mar (leap year: 31+29+31)
    vals = [10.0] * 91
    for i, d in enumerate(dates):
        if d.month == 2 and d.day > 10:                          # only 10 valid days in February
            vals[i] = None
    md = build_monthly_series(make_timeseries(dates, vals, resolution="daily"))
    assert md.values[pd.Timestamp("2020-01-01")] == 10.0
    assert np.isnan(md.values[pd.Timestamp("2020-02-01")])
    assert any("too few valid days" in n for n in md.notes)


def test_annual_anomaly_needs_nine_months_and_raw_mean_needs_twelve():
    ts = synthetic_monthly(start_year=2000, end_year=2002, noise_sd=0.1, seed=6)
    for i in range(12 + 8, 12 + 12):       # 2001: drop 4 months -> 8 valid
        ts.values[i] = None
    for i in range(24 + 10, 24 + 12):      # 2002: drop 2 months -> 10 valid
        ts.values[i] = None
    md = build_monthly_series(ts)
    an = monthly_anomalies(md.values)
    ann_anom, ann_mean = annual_series(md.values, an)
    assert not np.isnan(ann_anom[2000]) and np.isnan(ann_anom[2001]) and not np.isnan(ann_anom[2002])
    assert not np.isnan(ann_mean[2000]) and np.isnan(ann_mean[2001]) and np.isnan(ann_mean[2002])


def test_seasonal_cycle_is_removed_by_anomalies():
    ts = synthetic_monthly(slope_per_year=0.0, noise_sd=0.0, amp=10)
    an = monthly_anomalies(build_monthly_series(ts).values)
    assert np.allclose(an.to_numpy(), 0, atol=1e-9)
