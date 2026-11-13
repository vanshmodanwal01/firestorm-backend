"""
Data access for NASA Earth science variables.

Needs (install on your machine, not required for core.py):
    pip install xarray netcdf4 earthaccess cartopy

Workflow:
    1. Login once:  earthaccess.login()   (free NASA Earthdata account)
    2. Search + download monthly granules, OR pre-download to ./data/ before the hackathon
    3. open with xarray -> subset to bbox -> area-weighted mean -> pandas Series
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Variable catalog: plain-language definitions (the brief suggests this helps students)
# short_name / var are the usual GES DISC identifiers - verify on the dataset page.
# --------------------------------------------------------------------------
CATALOG = {
    "air_temperature": dict(
        name="Near-surface air temperature", units="K", source="MERRA-2",
        short_name="M2TMNXSLV", var="T2M",
        about="Temperature of the air ~2 m above ground. The most direct warming signal."),
    "precipitation": dict(
        name="Precipitation", units="mm/day", source="GPM IMERG Monthly",
        short_name="GPM_3IMERGM", var="precipitation",
        about="Rain + snow reaching the ground, estimated from satellite microwave sensors."),
    "soil_moisture": dict(
        name="Root-zone soil moisture", units="m3/m3", source="GLDAS Noah",
        short_name="GLDAS_NOAH025_M", var="SoilMoi0_10cm_inst",
        about="How wet the soil is near the surface. Key for drought and farming."),
    "evaporation": dict(
        name="Evapotranspiration", units="kg/m2/s", source="GLDAS Noah",
        short_name="GLDAS_NOAH025_M", var="Evap_tavg",
        about="Water returned to the air from soil and plants."),
    "ndvi": dict(
        name="Vegetation greenness (NDVI)", units="index", source="MODIS MOD13C2",
        short_name="MOD13C2", var="CMG 0.05 Deg Monthly NDVI",
        about="How green/dense vegetation is. Higher = healthier or more vegetation."),
    "aerosol": dict(
        name="Aerosol optical depth", units="unitless", source="MERRA-2",
        short_name="M2TMNXAER", var="TOTEXTTAU",
        about="How much sunlight dust/smoke/pollution blocks in the atmosphere."),
    "no2": dict(
        name="Tropospheric NO2", units="molec/cm2", source="OMI OMNO2d",
        short_name="OMNO2d", var="ColumnAmountNO2TropCloudScreened",
        about="Traffic/industry pollution gas; a marker of urban air quality."),
    "ozone": dict(
        name="Total column ozone", units="DU", source="OMI / AIRS",
        short_name="OMTO3d", var="ColumnAmountO3",
        about="Ozone in the whole atmosphere column; protects from UV."),
}

# Demo regions as (lon_min, lat_min, lon_max, lat_max)
REGIONS = {
    "Indo-Gangetic Plain": (74, 24, 88, 31),
    "Lucknow area": (80.5, 26.4, 81.4, 27.2),
    "Amazon Basin": (-70, -10, -52, 0),
    "Sahel": (-15, 12, 20, 17),
    "Aral Sea region": (58, 42, 62, 47),
    "Himalaya (western)": (74, 32, 80, 36),
}


# --------------------------------------------------------------------------
# Extraction helpers
# --------------------------------------------------------------------------
def _latlon_names(da):
    lat = next(c for c in da.coords if c.lower() in ("lat", "latitude"))
    lon = next(c for c in da.coords if c.lower() in ("lon", "longitude"))
    return lat, lon


def subset_bbox(da, bbox):
    """bbox = (lon_min, lat_min, lon_max, lat_max). Handles ascending/descending lat."""
    lat, lon = _latlon_names(da)
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_slice = slice(lat_min, lat_max) if da[lat][0] < da[lat][-1] else slice(lat_max, lat_min)
    return da.sel({lon: slice(lon_min, lon_max), lat: lat_slice})


def regional_series(da, bbox, scale=1.0, offset=0.0) -> pd.Series:
    """Area-weighted (cos latitude) regional mean -> monthly pandas Series."""
    sub = subset_bbox(da, bbox)
    lat, lon = _latlon_names(sub)
    w = np.cos(np.deg2rad(sub[lat]))
    mean = sub.weighted(w).mean(dim=[lat, lon], skipna=True)
    s = mean.to_series() * scale + offset
    s.index = pd.to_datetime(s.index.astype(str)) if not isinstance(s.index, pd.DatetimeIndex) else s.index
    s.index = s.index.to_period("M").to_timestamp()  # normalize to month start
    return s.sort_index()


def open_monthly(files, var):
    """Open many monthly NetCDF files as one DataArray with a time dimension."""
    import xarray as xr
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds[var]


def download_with_earthaccess(short_name, bbox, start, end, out_dir="data"):
    """Search + download monthly granules for a bbox/time range. Returns file paths."""
    import earthaccess
    earthaccess.login()
    results = earthaccess.search_data(short_name=short_name, bounding_box=bbox,
                                      temporal=(start, end))
    return earthaccess.download(results, out_dir)


def load_giovanni_csv(path) -> pd.Series:
    """Giovanni 'Time Series, Area-Averaged' CSV export.
    Skips the metadata lines at the top and keeps only rows that start with a date."""
    import re
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if re.match(r"^\d{4}-\d{2}-\d{2}", line.strip()):
                parts = line.strip().split(",")
                try:
                    rows.append((pd.to_datetime(parts[0][:10]), float(parts[1])))
                except (ValueError, IndexError):
                    continue
    s = pd.Series(dict(rows)).astype(float)
    s = s[s > -9000]                      # drop fill values
    s.index = s.index.to_period("M").to_timestamp()
    return s.sort_index()


# --------------------------------------------------------------------------
# Spatial trend map (opposite trends in different places)
# --------------------------------------------------------------------------
def trend_map(da, bbox=None, alpha=0.05):
    """Per-pixel linear trend of deseasonalized data, vectorized.
    Returns Dataset-like dict of 2D arrays: slope_per_decade, p_value, significant.
    Uses OLS (fast); confirm hotspots with Mann-Kendall via core.analyze_series."""
    import xarray as xr
    from scipy import stats
    if bbox is not None:
        da = subset_bbox(da, bbox)
    clim = da.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - clim
    t = (anom["time"].dt.year + (anom["time"].dt.month - 0.5) / 12).values
    y = anom.transpose("time", ...).values
    n = y.shape[0]
    ok = np.isfinite(y)
    cnt = ok.sum(0)
    tm = np.nanmean(np.where(ok, t[:, None, None], np.nan), axis=0)
    ym = np.nanmean(y, axis=0)
    dt = np.where(ok, t[:, None, None] - tm, 0)
    dy = np.where(ok, y - ym, 0)
    sxx = (dt ** 2).sum(0)
    slope = (dt * dy).sum(0) / np.where(sxx > 0, sxx, np.nan)
    resid = dy - slope * dt
    se = np.sqrt((resid ** 2).sum(0) / np.maximum(cnt - 2, 1) / np.where(sxx > 0, sxx, np.nan))
    tstat = slope / se
    p = 2 * (1 - stats.t.cdf(np.abs(tstat), np.maximum(cnt - 2, 1)))
    dims = [d for d in anom.dims if d != "time"]
    coords = {d: anom[d] for d in dims}
    out = xr.Dataset({
        "slope_per_decade": (dims, slope * 10),
        "p_value": (dims, p),
        "significant": (dims, p < alpha),
    }, coords=coords)
    return out


def plot_trend_map(trend_ds, title="", units="", path=None):
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lat, lon = _latlon_names(trend_ds["slope_per_decade"])
    sl = trend_ds["slope_per_decade"]
    vmax = float(np.nanpercentile(np.abs(sl.values), 98))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(trend_ds[lon], trend_ds[lat], sl.values, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
    # stipple NON-significant pixels
    ns = ~trend_ds["significant"].values
    yy, xx = np.meshgrid(trend_ds[lat], trend_ds[lon], indexing="ij")
    ax.scatter(xx[ns][::2], yy[ns][::2], s=1, c="k", alpha=.35)
    plt.colorbar(im, label=f"{units} / decade")
    ax.set_title(f"{title}\n(dots = not significant)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return fig