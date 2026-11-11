"""Full stack: HTTP request -> API -> trend service -> REAL PowerProvider parser + cache.

Only the network call to NASA is replaced. The mocked response copies the layout
confirmed by a live probe on 2026-11-10 (header.fill_value, parameters.<CODE>.units,
YYYYMM keys plus one YYYY13 annual row per year, a `messages` list). Its numbers
are SYNTHETIC with a known trend and are not NASA data.
"""
import httpx
from fastapi.testclient import TestClient

from app.api.dependencies import get_provider
from app.cache.db_cache import DbCache
from app.core.config import Settings
from app.main import create_app
from app.providers.power import PowerProvider
from app.utils.rate_limiter import SlidingWindowRateLimiter
from tests.helpers import synthetic_monthly

BODY = {"variable": "T2M", "latitude": 25.43, "longitude": 81.84, "start_year": 1985, "end_year": 2024}


def realistic_payload():
    ts = synthetic_monthly(1985, 2024, slope_per_year=0.03, seed=1)
    param = {f"{d.year}{d.month:02d}": round(v, 3) for d, v in zip(ts.dates, ts.values)}
    param["198501"] = -999.0                       # a missing month, as NASA marks it
    for y in range(1985, 2025):
        param[f"{y}13"] = 20.0                     # annual summary row
    return {
        "type": "Feature", "geometry": {"type": "Point", "coordinates": [81.84, 25.43, 80.0]},
        "properties": {"parameter": {"T2M": param}},
        "header": {"title": "NASA/POWER", "fill_value": -999.0},
        "messages": ["Synthetic test message from NASA"],
        "parameters": {"T2M": {"units": "C", "longname": "Temperature at 2 Meters"}},
        "times": {"data": 0.1},
    }


def test_full_stack_with_real_parser_cache_and_nasa_messages(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, json=realistic_payload())

    provider = PowerProvider(
        Settings(power_max_retries=1),
        cache=DbCache(f"sqlite:///{tmp_path}/t.db", 1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        limiter=SlidingWindowRateLimiter(1000, 1), backoff_base=0.01,
    )
    app = create_app()
    app.dependency_overrides[get_provider] = lambda: provider
    client = TestClient(app)

    r1 = client.post("/api/v1/trends/analyze", json=BODY)
    assert r1.status_code == 200, r1.text
    d = r1.json()["data"]
    assert d["trend"]["classification"] == "increasing"
    assert d["unit"] == "C"
    assert len(d["series"]["monthly_dates"]) == 479            # Jan 1985 was NASA fill value and trimmed
    assert not any("13" == k[4:] for k in map(str, d["series"]["monthly_dates"]))
    notes = " ".join(d["diagnostics"]["notes"])
    assert "annual-summary" in notes and "NASA message" in notes and "leading" in notes
    assert r1.json()["meta"]["cached"] is False

    r2 = client.post("/api/v1/trends/analyze", json=BODY)
    assert r2.json()["meta"]["cached"] is True and len(calls) == 1   # second call served from cache
    assert r2.json()["data"]["trend"] == d["trend"]
    q = dict(calls[0].params)
    assert q["parameters"] == "T2M" and q["community"] == "AG" and q["start"] == "1985"
