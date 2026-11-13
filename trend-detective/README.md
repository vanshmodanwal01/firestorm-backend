# Earth System Trend Detective - Data Science Module

## Files
- `core.py`  - stats engine: deseasonalize -> Sen's slope + CI -> Mann-Kendall (autocorrelation-corrected) -> OLS check -> plain-language verdict -> plots
- `data.py`  - variable catalog (definitions), demo regions, NASA data helpers (earthaccess/xarray), Giovanni CSV loader, per-pixel trend map
- `demo.py`  - offline demo with synthetic data + `analyze()` function (JSON-ready) for backend/frontend teammates

## Quick start
    pip install -r requirements.txt
    python demo.py            # works offline, writes PNGs to outputs/

## Real data (3 routes, easiest first)
1. **Giovanni CSV**: giovanni.gsfc.nasa.gov -> Time Series, Area-Averaged -> pick variable + box + 2000-2023
   -> download CSV -> `s = data.load_giovanni_csv("file.csv"); demo.analyze(s, "Precip", "Lucknow", "mm/day")`
2. **earthaccess + xarray** (needs free Earthdata login):
   ```python
   import data, demo
   files = data.download_with_earthaccess("M2TMNXSLV", data.REGIONS["Indo-Gangetic Plain"], "2000-01-01", "2023-12-31")
   da = data.open_monthly(files, "T2M")
   s = data.regional_series(da, data.REGIONS["Indo-Gangetic Plain"], offset=-273.15)   # K -> C
   out = demo.analyze(s, "Air temperature", "Indo-Gangetic Plain", "deg C", plot_path="t2m.png")
   ```
3. **Pre-download** before Nov 14 (recommended backup for the hackathon).

## Trend map (opposite trends in different regions)
    tm = data.trend_map(da, bbox)
    data.plot_trend_map(tm, "T2M trend", "K", path="map.png")

## Methods (for the write-up / judges)
- Monthly data, seasonal cycle removed (monthly climatology subtracted)
- Rate: Theil-Sen slope per decade with 95% CI (robust to outliers)
- Significance: Mann-Kendall, variance corrected for lag-1 autocorrelation; also OLS with effective sample size
- "Significant" = MK p < 0.05 AND Sen CI excludes zero
- Validated: MK matches scipy Kendall tau; ~4.5% false positives on trendless AR(1) noise, 93% detection of a 0.5/decade trend
- Limits: linear trends only; not for non-linear/step changes; short records (<10 yrs) have low power
