"""End-to-end trend analysis of one variable at one location.

Pipeline: NASA data -> monthly series -> monthly anomalies (seasonal cycle
removed) -> annual anomalies -> trend tests -> classification -> insight.

Primary inference runs on ANNUAL anomalies (one value per year), because monthly
values are strongly autocorrelated. The test used is Mann-Kendall with the
Hamed-Rao autocorrelation correction; Sen's slope gives the rate and its
interval; OLS with robust errors is a cross-check. A seasonal Mann-Kendall on
the raw monthly values is reported as supporting evidence only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.algorithms.mann_kendall import hamed_rao_mann_kendall, seasonal_mann_kendall
from app.algorithms.regression import ols_trend
from app.algorithms.sen_slope import seasonal_sens_slope, sens_slope
from app.algorithms.significance import MIN_YEARS_FOR_TREND, classify_trend, effect_size
from app.core.errors import ValidationFailed
from app.providers.base import DataProvider
from app.providers.variables import VARIABLES, VariableInfo, get_variable
from app.schemas.timeseries import TimeSeries
from app.schemas.trends import (
    ChangeOut, Diagnostics, Location, MannKendallOut, OlsOut, SenSlopeOut, SeasonalMkOut,
    SeriesBlock, TrendAnalysisData, TrendRequest, TrendResult,
)
from app.schemas.variables import VariableOut
from app.services.insight_service import generate_insight
from app.services.preprocessing_service import (
    MIN_VALID_MONTHS_PER_YEAR, annual_series, build_monthly_series, monthly_anomalies,
)
from app.utils.dates import provider_dates

BASELINE_NOTE = (
    "Anomalies are measured against each calendar month's average over the analysed period, "
    "so they show change within the period, not departure from a long-term climate normal."
)


def variable_out(info: VariableInfo) -> VariableOut:
    return VariableOut(
        code=info.code, name=info.name, definition=info.definition,
        default_unit=info.fallback_unit, category=info.category,
        is_reanalysis=info.is_reanalysis, dataset_note=info.dataset_note,
        coverage_start_year=info.coverage_start_year,
        percent_change_meaningful=info.ratio_scale,
    )


def _clean(values) -> list[float | None]:
    out: list[float | None] = []
    for v in values:
        out.append(None if v is None or not np.isfinite(v) else round(float(v), 6))
    return out


def _empty_trend(alpha: float, reason: str) -> TrendResult:
    return TrendResult(
        classification="insufficient_data", direction=None, significant=False, alpha=alpha,
        primary_method="none", reasons=[reason],
    )


def analyze_timeseries(ts: TimeSeries, info: VariableInfo, alpha: float = 0.05) -> TrendAnalysisData:
    md = build_monthly_series(ts)
    monthly = md.values
    notes = list(ts.quality.notes) + list(md.notes)

    anomalies = monthly_anomalies(monthly)
    ann_anom, ann_mean = annual_series(monthly, anomalies)

    n_total = len(monthly)
    n_valid = int(monthly.notna().sum())
    missing_fraction = round((n_total - n_valid) / n_total, 6) if n_total else 1.0

    valid = ann_anom.dropna()
    n_years = len(valid)
    lag1: float | None = None
    sen = None
    trend: TrendResult

    if n_years < MIN_YEARS_FOR_TREND:
        trend = _empty_trend(
            alpha,
            f"Only {n_years} year(s) have enough valid months (at least {MIN_YEARS_FOR_TREND} years are "
            f"needed, and each year needs {MIN_VALID_MONTHS_PER_YEAR} valid months).",
        )
    else:
        x = valid.index.to_numpy(dtype=float)
        y = valid.to_numpy(dtype=float)
        n_range = int(x.max() - x.min()) + 1
        if n_range > n_years:
            notes.append(
                f"{n_range - n_years} year(s) inside the period lacked enough valid months and were "
                "left out of the trend test"
            )

        mk = hamed_rao_mann_kendall(y, alpha)
        sen = sens_slope(x, y, var_s=mk.var_s, alpha=alpha)
        ols = ols_trend(x, y, alpha)
        assess = classify_trend(mk.p_value, mk.s, sen.slope, ols.slope, alpha)
        reasons = list(assess.reasons)
        if assess.significant and ols.p_value_hac >= alpha:
            reasons.append(
                f"Linear regression with robust errors was less certain (p = {ols.p_value_hac:.3g})."
            )

        # Supporting evidence: seasonal Mann-Kendall on raw monthly values.
        seasonal_out = None
        matrix = (
            monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
            .reindex(columns=range(1, 13))
        )
        if len(matrix) >= MIN_YEARS_FOR_TREND:
            try:
                smk = seasonal_mann_kendall(matrix.to_numpy(), alpha)
                sslope = seasonal_sens_slope(matrix.index.to_numpy(dtype=float), matrix.to_numpy())
                s_sig = smk.p_value < alpha
                agrees = (s_sig == assess.significant) and (not s_sig or np.sign(smk.z) == np.sign(mk.z))
                seasonal_out = SeasonalMkOut(
                    p_value=smk.p_value, z=smk.z,
                    slope_per_decade=None if sslope is None else sslope * 10,
                    n_seasons_used=smk.n_seasons or 0, agrees_with_primary=bool(agrees),
                )
            except ValueError:
                notes.append("The seasonal test could not be run (too few values in every calendar month)")

        span = float(x.max() - x.min())
        total = sen.slope * span
        resid = y - (sen.intercept + sen.slope * x)
        es, label = effect_size(total, float(resid.std(ddof=1)))

        pct, pct_note = None, None
        if not info.ratio_scale:
            pct_note = "Percent change is not reported because this scale has no true zero (e.g. °C)."
        else:
            baseline = float(monthly.mean())
            if abs(baseline) > 1e-9:
                pct = total / baseline * 100.0
            else:
                pct_note = "The period average is zero, so percent change is undefined."

        if np.std(resid[:-1]) > 0 and np.std(resid[1:]) > 0:
            lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])

        unit = ts.unit
        trend = TrendResult(
            classification=assess.classification, direction=assess.direction,
            significant=assess.significant, alpha=alpha, primary_method=mk.method,
            mann_kendall=MannKendallOut(
                method=mk.method, n=mk.n, tau=mk.tau, s=mk.s, z=mk.z, p_value=mk.p_value,
                p_value_uncorrected=mk.p_value_uncorrected,
                variance_inflation_factor=mk.variance_inflation,
            ),
            sen_slope=SenSlopeOut(
                slope_per_decade=sen.slope * 10, ci_low_per_decade=sen.ci_low * 10,
                ci_high_per_decade=sen.ci_high * 10, confidence_level=sen.confidence_level,
                unit_per_decade=unit,
            ),
            ols=OlsOut(
                slope_per_decade=ols.slope * 10, ci_low_per_decade=ols.ci_low * 10,
                ci_high_per_decade=ols.ci_high * 10, p_value=ols.p_value_hac,
                p_value_naive=ols.p_value_naive, r_squared=ols.r_squared,
                standard_error_method=f"Newey-West HAC, {ols.hac_lags} lag(s), small-sample corrected",
            ),
            seasonal_mann_kendall=seasonal_out,
            change=ChangeOut(
                period_start_year=int(x.min()), period_end_year=int(x.max()), total_change=total,
                unit=unit, pct_change=pct, pct_change_note=pct_note, effect_size=es,
                effect_size_label=label,
            ),
            reasons=reasons,
        )

    # ---- chart-ready series -------------------------------------------------
    if n_total:
        years = list(range(int(monthly.index.year.min()), int(monthly.index.year.max()) + 1))
    else:
        years = []
    ann_anom_r = ann_anom.reindex(years) if years else pd.Series(dtype=float)
    ann_mean_r = ann_mean.reindex(years) if years else pd.Series(dtype=float)
    ma12 = anomalies.rolling(12, center=True, min_periods=12).mean()
    ma5 = ann_anom_r.rolling(5, center=True, min_periods=5).mean() if years else ann_anom_r
    if sen is not None:
        line = [sen.intercept + sen.slope * yr for yr in years]
    else:
        line = [None] * len(years)

    series = SeriesBlock(
        monthly_dates=[d.date() for d in monthly.index],
        monthly_values=_clean(monthly.to_numpy()),
        monthly_anomalies=_clean(anomalies.to_numpy()),
        monthly_anomalies_moving_avg_12m=_clean(ma12.to_numpy()),
        annual_years=years,
        annual_anomaly=_clean(ann_anom_r.to_numpy()),
        annual_mean=_clean(ann_mean_r.to_numpy()),
        annual_anomaly_moving_avg_5y=_clean(ma5.to_numpy()),
        trend_line=_clean(line),
    )

    diag = Diagnostics(
        n_years_used=n_years, n_months_total=n_total, n_months_valid=n_valid,
        missing_fraction=missing_fraction, lag1_autocorrelation=lag1, baseline=BASELINE_NOTE,
        outliers_flagged=len(md.outlier_dates),
        outlier_dates=[d.date() for d in md.outlier_dates[:50]],
        notes=notes,
    )
    insight = generate_insight(info, ts.unit, ts.latitude, ts.longitude, trend, diag, ts.provenance)

    return TrendAnalysisData(
        variable=variable_out(info), unit=ts.unit,
        location=Location(latitude=ts.latitude, longitude=ts.longitude),
        series=series, trend=trend, diagnostics=diag, insight=insight, provenance=ts.provenance,
    )


class TrendAnalysisService:
    def __init__(self, provider: DataProvider):
        self.provider = provider

    async def analyze(self, req: TrendRequest) -> TrendAnalysisData:
        info = get_variable(req.variable)
        if info is None:
            raise ValidationFailed(
                f"unsupported variable '{req.variable}'", {"supported": sorted(VARIABLES)}
            )
        start, end = provider_dates(req.start_year, req.end_year, req.resolution)
        result = await self.provider.fetch_point(
            [info.code], req.latitude, req.longitude, start, end, req.resolution
        )
        return analyze_timeseries(result[info.code], info, req.alpha)
