"""Run this on YOUR machine (needs internet) to check the real NASA POWER response.

    cd backend
    python scripts/probe_power.py --lat 25.43 --lon 81.84 --start 2020 --end 2021

It saves the raw JSON to power_probe.json and prints the structure the parser
depends on. Paste the output back so the parser can be confirmed against real
data. No API key is needed.
"""
import argparse
import json

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, default=25.43)
ap.add_argument("--lon", type=float, default=81.84)
ap.add_argument("--start", default="2020")
ap.add_argument("--end", default="2021")
ap.add_argument("--vars", default="T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN")
a = ap.parse_args()

url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
params = dict(parameters=a.vars, community="AG", longitude=a.lon, latitude=a.lat,
              start=a.start, end=a.end, format="JSON")
r = httpx.get(url, params=params, timeout=60)
print("HTTP", r.status_code)
data = r.json()
with open("power_probe.json", "w") as f:
    json.dump(data, f, indent=2)

print("top-level keys:", list(data))
print("header keys:", list(data.get("header", {})))
print("header.fill_value:", data.get("header", {}).get("fill_value"))
print("parameters block:", json.dumps(data.get("parameters"), indent=2)[:800])
for code, series in data.get("properties", {}).get("parameter", {}).items():
    keys = list(series)
    print(code, "->", len(keys), "keys; first 3:", keys[:3], "last 3:", keys[-3:])
