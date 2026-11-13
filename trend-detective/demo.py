"""
Offline demo (no internet / no NASA login needed).
Run:  python demo.py
Creates synthetic monthly series that mimic real behavior so you can test the pipeline
and show examples. Replace with real data via analyze() below.
"""
import os
import numpy as np
import pandas as pd
from core import analyze_series, plot_trend

OUT = "outputs"


def analyze(series, variable, region, units="", plot_path=None):
    """THE function your frontend/backend teammates call.
    series: monthly pd.Series. Returns dict (JSON-ready) and optionally saves a PNG."""
    result, df = analyze_series(series, variable, region, units)
    if plot_path:
        plot_trend(result, df, plot_path)
    d = result.to_dict()
    d["timeseries"] = {
        "dates": [x.strftime("%Y-%m") for x in df.index],
        "raw": df["raw"].round(4).tolist(),
        "anomaly": df["anomaly"].round(4).tolist(),
    }
    return d


def synth(n_years=24, base=25, amp=8, trend_per_decade=0.0, noise=1.0, ar=0.5, seed=0, start="2000-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_years * 12, freq="MS")
    t = np.arange(len(idx)) / 12
    season = amp * np.sin(2 * np.pi * (idx.month - 4) / 12)
    e = np.zeros(len(idx))
    for i in range(1, len(idx)):                      # autocorrelated noise, like real climate data
        e[i] = ar * e[i - 1] + rng.normal(0, noise)
    return pd.Series(base + season + trend_per_decade * (t / 10) + e, index=idx)


DEMOS = [
    # (variable, region, units, kwargs)
    ("Near-surface air temperature", "Indo-Gangetic Plain", "deg C",
     dict(trend_per_decade=0.35, noise=0.8, seed=1)),             # clear warming
    ("Precipitation", "Sahel", "mm/day",
     dict(base=3, amp=2.5, trend_per_decade=0.0, noise=0.6, seed=2)),   # no trend
    ("Vegetation greenness (NDVI)", "Aral Sea region", "index",
     dict(base=0.4, amp=0.15, trend_per_decade=-0.04, noise=0.03, seed=3)),  # decline
]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for var, reg, units, kw in DEMOS:
        s = synth(**kw)
        res = analyze(s, var, reg, units, plot_path=f"{OUT}/{reg.split()[0].lower()}_{var.split()[0].lower()}.png")
        print("-", res["summary"])
        print(f"  Sen slope/decade={res['slope_per_decade']:+.3f}  MK p={res['mk_p']:.4f}  OLS p={res['ols_p']:.4f}  significant={res['significant']}\n")
