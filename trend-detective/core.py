"""
Earth System Trend Detective - core statistics.

Input : a monthly pandas Series (DatetimeIndex) for ONE variable in ONE region
Output: rate of change, significance (OLS + Mann-Kendall/Sen), plain-language verdict, plots

Only needs numpy, scipy, pandas, matplotlib.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# --------------------------------------------------------------------------
# 1. Preprocessing
# --------------------------------------------------------------------------
def deseasonalize(s: pd.Series) -> pd.Series:
    """Subtract the mean monthly climatology -> anomalies (removes seasonal cycle)."""
    clim = s.groupby(s.index.month).transform("mean")
    return s - clim


def to_decimal_years(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(idx.year, float) + (np.asarray(idx.dayofyear, float) - 0.5) / np.where(idx.is_leap_year, 366, 365)


def lag1_autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    denom = np.sum(x * x)
    return float(np.sum(x[1:] * x[:-1]) / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------
# 2. Statistical tests
# --------------------------------------------------------------------------
def linear_trend(t: np.ndarray, y: np.ndarray) -> dict:
    """OLS slope with 95% CI. p-value uses an effective sample size to account
    for autocorrelation (monthly data are NOT independent)."""
    res = stats.linregress(t, y)
    n = len(y)
    resid = y - (res.intercept + res.slope * t)
    r1 = max(lag1_autocorr(resid), 0.0)
    n_eff = max(n * (1 - r1) / (1 + r1), 3.0)
    se = res.stderr * np.sqrt((n - 2) / (n_eff - 2)) if n_eff > 2 else res.stderr
    tcrit = stats.t.ppf(1 - ALPHA / 2, n_eff - 2)
    p = 2 * (1 - stats.t.cdf(abs(res.slope / se), n_eff - 2))
    return dict(slope=res.slope, intercept=res.intercept, r2=res.rvalue ** 2,
                p_value=float(p), ci_low=res.slope - tcrit * se,
                ci_high=res.slope + tcrit * se, n_eff=float(n_eff), lag1=r1)


def mann_kendall(y: np.ndarray, var_inflation: float = 1.0) -> dict:
    """Mann-Kendall test (tie-corrected) on a series ordered in time.
    var_inflation = n/n_eff corrects the variance for autocorrelation (>=1).
    Returns S, Z, two-sided p, Kendall tau and trend direction."""
    y = np.asarray(y, float)
    n = len(y)
    diff = y[None, :] - y[:, None]          # y_j - y_i
    s = np.sum(np.sign(np.triu(diff, k=1)))
    _, counts = np.unique(y, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    var_s *= var_inflation
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return dict(S=float(s), Z=float(z), p_value=float(p), tau=float(tau),
                direction="increasing" if z > 0 else "decreasing" if z < 0 else "none")


def sens_slope(t: np.ndarray, y: np.ndarray, var_inflation: float = 1.0) -> dict:
    """Theil-Sen slope (median of all pairwise slopes) with 95% CI (Gilbert 1987)."""
    n = len(y)
    i, j = np.triu_indices(n, k=1)
    dt = t[j] - t[i]
    ok = dt != 0
    slopes = (y[j] - y[i])[ok] / dt[ok]
    slopes.sort()
    m = len(slopes)
    med = float(np.median(slopes))
    var_s = n * (n - 1) * (2 * n + 5) / 18 * var_inflation
    c = stats.norm.ppf(1 - ALPHA / 2) * np.sqrt(var_s)
    lo = int(np.clip(np.floor((m - c) / 2), 0, m - 1))
    hi = int(np.clip(np.ceil((m + c) / 2), 0, m - 1))
    return dict(slope=med, ci_low=float(slopes[lo]), ci_high=float(slopes[hi]))


# --------------------------------------------------------------------------
# 3. One-call analysis
# --------------------------------------------------------------------------
@dataclass
class TrendResult:
    variable: str
    region: str
    units: str
    start: str
    end: str
    n_months: int
    slope_per_decade: float          # Sen's slope, in units / decade
    slope_ci_low: float
    slope_ci_high: float
    ols_slope_per_decade: float
    ols_p: float
    mk_p: float
    mk_tau: float
    lag1_autocorr: float
    significant: bool
    direction: str                   # increasing / decreasing / no clear trend
    pct_change_per_decade: float | None
    summary: str

    def to_dict(self):
        return asdict(self)


def analyze_series(series: pd.Series, variable="variable", region="region",
                   units="", deseason=True) -> tuple[TrendResult, pd.DataFrame]:
    """Full pipeline for one monthly series. Returns (result, dataframe for plotting)."""
    s = series.dropna().sort_index()
    if len(s) < 36:
        raise ValueError("Need at least ~3 years (36 monthly values) for a trend.")
    anom = deseasonalize(s) if deseason else s - s.mean()
    t = to_decimal_years(s.index)

    ols = linear_trend(t, anom.values)
    # autocorrelation correction: detrend with a quick Sen slope, take lag-1 of residuals
    raw_sen = sens_slope(t, anom.values)
    resid = anom.values - raw_sen["slope"] * t
    r1 = min(max(lag1_autocorr(resid), 0.0), 0.9)
    infl = (1 + r1) / (1 - r1)
    mk = mann_kendall(anom.values, var_inflation=infl)
    sen = sens_slope(t, anom.values, var_inflation=infl)

    # significant only if MK says so AND the Sen CI excludes zero
    significant = (mk["p_value"] < ALPHA) and (sen["ci_low"] > 0 or sen["ci_high"] < 0)
    direction = ("increasing" if sen["slope"] > 0 else "decreasing") if significant else "no clear trend"
    per_dec = sen["slope"] * 10
    mean_val = float(s.mean())
    pct = per_dec / abs(mean_val) * 100 if abs(mean_val) > 1e-12 else None

    start, end = s.index[0].strftime("%Y-%m"), s.index[-1].strftime("%Y-%m")
    u = f" {units}" if units else ""
    if significant:
        summary = (f"{variable} over {region} shows a statistically significant {direction} trend "
                   f"({start} to {end}): about {per_dec:+.3g}{u} per decade "
                   f"(95% CI {sen['ci_low']*10:+.3g} to {sen['ci_high']*10:+.3g})"
                   + (f", roughly {pct:+.1f}% of the average per decade." if pct is not None else "."))
    else:
        summary = (f"{variable} over {region} shows a {per_dec:+.3g}{u} per decade tendency "
                   f"({start} to {end}), but it is NOT statistically significant "
                   f"(Mann-Kendall p = {mk['p_value']:.2f}). A longer record might change this.")

    result = TrendResult(variable, region, units, start, end, len(s), per_dec,
                         sen["ci_low"] * 10, sen["ci_high"] * 10, ols["slope"] * 10,
                         ols["p_value"], mk["p_value"], mk["tau"], float(r1), bool(significant),
                         direction, pct, summary)
    df = pd.DataFrame({"raw": s.values, "anomaly": anom.values, "t": t}, index=s.index)
    df.attrs.update(dict(sen=sen, ols=ols))
    return result, df


# --------------------------------------------------------------------------
# 4. Plotting
# --------------------------------------------------------------------------
def plot_trend(result: TrendResult, df: pd.DataFrame, path: str | None = None):
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sen = df.attrs["sen"]
    t = df["t"].values
    med_t = np.median(t)
    # Sen line passes through median of anomalies at median time
    line = sen["slope"] * (t - med_t) + np.median(df["anomaly"] - sen["slope"] * (t - med_t))
    lo = sen["ci_low"] * (t - med_t) + np.median(df["anomaly"] - sen["ci_low"] * (t - med_t))
    hi = sen["ci_high"] * (t - med_t) + np.median(df["anomaly"] - sen["ci_high"] * (t - med_t))

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(df.index, df["raw"], color="#4c78a8", lw=1)
    ax[0].set_title(f"{result.variable} - {result.region}: raw monthly data")
    ax[0].set_ylabel(result.units)
    ax[0].grid(alpha=.3)

    ax[1].plot(df.index, df["anomaly"], color="#999", lw=1, label="Deseasonalized anomaly")
    ax[1].fill_between(df.index, lo, hi, color="#e45756", alpha=.2, label="95% CI of slope")
    col = "#e45756" if result.significant else "#7f7f7f"
    ax[1].plot(df.index, line, color=col, lw=2.5,
               label=f"Sen's slope {result.slope_per_decade:+.3g} {result.units}/decade"
                     + ("  (significant)" if result.significant else "  (not significant)"))
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_ylabel(f"Anomaly ({result.units})")
    ax[1].legend(loc="best", fontsize=9)
    ax[1].grid(alpha=.3)
    fig.suptitle(f"Mann-Kendall p = {result.mk_p:.3g}   |   OLS p = {result.ols_p:.3g}", y=0.995, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
