"""Request/response schemas for trend analysis (the contract the frontend consumes)."""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.timeseries import Provenance
from app.schemas.variables import VariableOut

Classification = Literal[
    "increasing", "decreasing", "no_significant_trend", "inconclusive", "insufficient_data"
]


class TrendRequest(BaseModel):
    variable: str = Field(examples=["T2M"])
    latitude: float = Field(ge=-90, le=90, examples=[25.43])
    longitude: float = Field(ge=-180, le=180, examples=[81.84])
    start_year: int = Field(ge=1981, examples=[1985])
    end_year: int = Field(ge=1981, examples=[2025])
    resolution: Literal["monthly", "daily"] = Field(
        "monthly", description="Daily data are averaged to monthly means before analysis"
    )
    alpha: float = Field(0.05, ge=0.001, le=0.1, description="Significance level")


class Location(BaseModel):
    latitude: float
    longitude: float


class SeriesBlock(BaseModel):
    """Parallel arrays, ready to plot. null marks a missing value."""

    monthly_dates: list[date]
    monthly_values: list[float | None]
    monthly_anomalies: list[float | None] = Field(
        description="Value minus the average for that calendar month over the analysed period"
    )
    monthly_anomalies_moving_avg_12m: list[float | None] = Field(
        description="Centered 12-month moving average of the monthly anomalies (descriptive only)"
    )
    annual_years: list[int]
    annual_anomaly: list[float | None] = Field(
        description="Mean of monthly anomalies per year; null if fewer than 9 valid months"
    )
    annual_mean: list[float | None] = Field(
        description="Mean of raw values per year; null unless all 12 months are valid"
    )
    annual_anomaly_moving_avg_5y: list[float | None] = Field(
        description="Centered 5-year moving average (descriptive only; first/last two years are null)"
    )
    trend_line: list[float | None] = Field(
        description="Sen's-slope line on the annual-anomaly scale, one value per annual_years entry"
    )


class MannKendallOut(BaseModel):
    method: str
    n: int
    tau: float
    s: float
    z: float
    p_value: float = Field(description="Two-sided p-value after the autocorrelation correction")
    p_value_uncorrected: float = Field(description="Classic Mann-Kendall p-value, for comparison")
    variance_inflation_factor: float = Field(description="1.0 means no autocorrelation correction applied")


class SenSlopeOut(BaseModel):
    slope_per_decade: float
    ci_low_per_decade: float
    ci_high_per_decade: float
    confidence_level: float
    unit_per_decade: str


class OlsOut(BaseModel):
    slope_per_decade: float
    ci_low_per_decade: float
    ci_high_per_decade: float
    p_value: float = Field(description="Autocorrelation-robust (Newey-West) p-value")
    p_value_naive: float = Field(description="Plain OLS p-value, for comparison only")
    r_squared: float
    standard_error_method: str


class SeasonalMkOut(BaseModel):
    p_value: float
    z: float
    slope_per_decade: float | None
    n_seasons_used: int
    agrees_with_primary: bool


class ChangeOut(BaseModel):
    period_start_year: int
    period_end_year: int
    total_change: float = Field(description="Sen slope times the length of the period, in the variable's unit")
    unit: str
    pct_change: float | None = Field(description="Total change as % of the period mean; null where meaningless")
    pct_change_note: str | None = None
    effect_size: float | None = Field(description="|total change| / year-to-year variation around the trend")
    effect_size_label: Literal["small", "moderate", "large"] | None = None


class TrendResult(BaseModel):
    classification: Classification
    direction: Literal["increasing", "decreasing"] | None
    significant: bool
    alpha: float
    primary_method: str
    mann_kendall: MannKendallOut | None = None
    sen_slope: SenSlopeOut | None = None
    ols: OlsOut | None = None
    seasonal_mann_kendall: SeasonalMkOut | None = None
    change: ChangeOut | None = None
    reasons: list[str] = Field(default_factory=list)


class Diagnostics(BaseModel):
    n_years_used: int
    n_months_total: int
    n_months_valid: int
    missing_fraction: float
    lag1_autocorrelation: float | None = Field(
        description="Lag-1 autocorrelation of annual residuals around the trend line"
    )
    baseline: str
    outliers_flagged: int = Field(description="Months flagged as unusual for review; all were KEPT")
    outlier_dates: list[date]
    notes: list[str]


class TrendAnalysisData(BaseModel):
    variable: VariableOut
    unit: str
    location: Location
    series: SeriesBlock
    trend: TrendResult
    diagnostics: Diagnostics
    insight: str
    provenance: Provenance
