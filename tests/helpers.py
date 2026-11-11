"""Synthetic fixtures for tests.

Everything here is SYNTHETIC and labeled as such. It exists only so tests can
check logic with a KNOWN true trend; it is never served as NASA data.
"""
from datetime import date

import numpy as np

from app.schemas.timeseries import DataQuality, Provenance, TimeSeries


def make_timeseries(dates, values, resolution="monthly", variable="T2M", unit="C",
                    lat=25.43, lon=81.84) -> TimeSeries:
    n_valid = sum(v is not None for v in values)
    n = len(values)
    return TimeSeries(
        variable=variable, unit=unit, unit_from_upstream=True, temporal_resolution=resolution,
        latitude=lat, longitude=lon, dates=dates, values=values,
        quality=DataQuality(n_total=n, n_valid=n_valid, n_missing=n - n_valid,
                            missing_fraction=0 if n == 0 else (n - n_valid) / n, notes=[]),
        provenance=Provenance(source="TEST FIXTURE (synthetic, not NASA data)",
                              dataset_note="synthetic", is_reanalysis=True),
    )


def synthetic_monthly(start_year=1985, end_year=2024, slope_per_year=0.03, noise_sd=0.5, seed=1,
                      base=20.0, amp=8.0, missing_frac=0.0, variable="T2M", unit="C",
                      lat=25.43, lon=81.84) -> TimeSeries:
    rng = np.random.default_rng(seed)
    dates, vals = [], []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            t = (y - start_year) + (m - 1) / 12
            v = base + amp * np.sin(2 * np.pi * (m - 4) / 12) + slope_per_year * t + rng.normal(0, noise_sd)
            dates.append(date(y, m, 1))
            vals.append(float(v))
    if missing_frac:
        drop = rng.random(len(vals)) < missing_frac
        vals = [None if d else v for v, d in zip(vals, drop)]
    return make_timeseries(dates, vals, variable=variable, unit=unit, lat=lat, lon=lon)
