"""Cleaning and aggregation of a TimeSeries into analysis-ready monthly data.

Principle: NEVER silently modify or discard scientific observations. Every
action that changes the data (de-duplicating, inserting missing months,
trimming empty edges, converting daily to monthly) is recorded in `notes`.
Outliers are only FLAGGED for review; they stay in the data, because real
extreme events are exactly what an Earth-system analysis should keep.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.schemas.timeseries import TimeSeries

MIN_VALID_DAYS_PER_MONTH = 20     # daily -> monthly mean needs at least this many valid days
MIN_VALID_MONTHS_PER_YEAR = 9     # annual anomaly needs at least this many valid months
OUTLIER_ROBUST_Z = 5.0            # robust (MAD-based) z-score above which a month is flagged


@dataclass
class MonthlyData:
    values: pd.Series                       # DatetimeIndex (month start), float with NaN
    notes: list[str] = field(default_factory=list)
    outlier_dates: list[pd.Timestamp] = field(default_factory=list)


def monthly_anomalies(values: pd.Series) -> pd.Series:
    """Value minus the mean of the same calendar month over the whole series."""
    if values.empty:
        return values.copy()
    return values - values.groupby(values.index.month).transform("mean")


def flag_outliers(anomalies: pd.Series, threshold: float = OUTLIER_ROBUST_Z) -> list[pd.Timestamp]:
    a = anomalies.dropna()
    if len(a) < 24:
        return []
    med = a.median()
    mad = (a - med).abs().median()
    if mad == 0:
        return []
    z = 0.6745 * (a - med) / mad
    return list(a.index[z.abs() > threshold])


def annual_series(values: pd.Series, anomalies: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (annual anomaly, annual raw mean), both indexed by integer year.

    Annual anomaly = mean of monthly anomalies, needs >= 9 valid months, so a
    partially observed year is not biased towards the months it happens to have.
    Annual raw mean is only given for years with all 12 months valid.
    """
    if values.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    year = values.index.year
    n_anom = anomalies.groupby(year).count()
    ann_anom = anomalies.groupby(year).mean().where(n_anom >= MIN_VALID_MONTHS_PER_YEAR)
    n_raw = values.groupby(year).count()
    ann_mean = values.groupby(year).mean().where(n_raw == 12)
    return ann_anom, ann_mean


def _daily_to_monthly(s: pd.Series) -> tuple[pd.Series, int]:
    g = s.groupby(s.index.to_period("M"))
    valid = g.count()
    mean = g.mean().where(valid >= MIN_VALID_DAYS_PER_MONTH)
    n_excluded = int((valid < MIN_VALID_DAYS_PER_MONTH).sum())
    mean.index = mean.index.to_timestamp()
    return mean, n_excluded


def build_monthly_series(ts: TimeSeries) -> MonthlyData:
    notes: list[str] = []
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if not ts.dates:
        return MonthlyData(empty, ["The data source returned no observations."])

    s = pd.Series(
        [np.nan if v is None else float(v) for v in ts.values],
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in ts.dates]),
        dtype=float,
    )

    if ts.temporal_resolution == "daily":
        s = s.sort_index()
        dup = s.index.duplicated(keep="first")
        if dup.any():
            notes.append(f"{int(dup.sum())} duplicate date(s) found; the first occurrence of each was kept")
            s = s[~dup]
        s, n_excl = _daily_to_monthly(s)
        notes.append(
            f"Daily values were averaged into monthly means (a month needs at least "
            f"{MIN_VALID_DAYS_PER_MONTH} valid days)."
        )
        if n_excl:
            notes.append(f"{n_excl} month(s) had too few valid days and were set to missing")
    else:
        s.index = s.index.to_period("M").to_timestamp()
        s = s.sort_index()
        dup = s.index.duplicated(keep="first")
        if dup.any():
            notes.append(f"{int(dup.sum())} duplicate month(s) found; the first occurrence of each was kept")
            s = s[~dup]

    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    n_gaps = len(full) - len(s)
    if n_gaps:
        notes.append(f"{n_gaps} month(s) absent from the data were inserted as missing (not filled in)")
    s = s.reindex(full)

    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None:
        notes.append("No valid observations were found in the requested period.")
        return MonthlyData(s.iloc[0:0], notes)
    lead = int((s.index < first).sum())
    trail = int((s.index > last).sum())
    s = s.loc[first:last]
    if lead:
        notes.append(f"{lead} leading month(s) without data were trimmed; usable data begin {first:%Y-%m}")
    if trail:
        notes.append(f"{trail} trailing month(s) without data were trimmed; usable data end {last:%Y-%m}")

    outliers = flag_outliers(monthly_anomalies(s))
    if outliers:
        notes.append(
            f"{len(outliers)} month(s) were flagged as statistically unusual for review; "
            "all were KEPT in the analysis"
        )
    return MonthlyData(s, notes, outliers)
