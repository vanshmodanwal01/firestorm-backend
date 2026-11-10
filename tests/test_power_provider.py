"""Offline tests for the POWER provider.

The FIXTURE below is SYNTHETIC: it is shaped like the documented POWER JSON so
we can test parsing logic without network access. It is NOT NASA data and must
never be served as such. Real-format confirmation: scripts/probe_power.py.
"""
import asyncio
import time

import httpx
import pytest

from app.cache.db_cache import DbCache
from app.core.config import Settings
from app.core.errors import (
    NoDataAvailable,
    UpstreamBadResponse,
    UpstreamUnavailable,
    ValidationFailed,
)
from app.providers.power import PowerProvider
from app.utils.rate_limiter import SlidingWindowRateLimiter

SYNTHETIC_PAYLOAD = {
    "header": {"fill_value": -999.0},
    "properties": {
        "parameter": {
            "T2M": {
                "202001": 14.5,
                "202002": 17.0,
                "202003": -999.0,
                "202013": 15.0,  # annual-summary row
            }
        }
    },
    "parameters": {"T2M": {"units": "C", "longname": "Temperature at 2 Meters"}},
}


def make_provider(handler, tmp_path=None, retries=3, limiter=None):
    settings = Settings(power_max_retries=retries, power_timeout_seconds=5)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cache = DbCache(f"sqlite:///{tmp_path}/c.db", ttl_hours=1) if tmp_path else None
    limiter = limiter or SlidingWindowRateLimiter(1000, 1)
    return PowerProvider(settings, cache=cache, client=client, limiter=limiter, backoff_base=0.01)


def run(coro):
    return asyncio.run(coro)


def fetch(p, **kw):
    args = dict(variables=["T2M"], latitude=25.43, longitude=81.84, start="2020", end="2020")
    args.update(kw)
    return run(p.fetch_point(**args))


def ok_handler(request):
    return httpx.Response(200, json=SYNTHETIC_PAYLOAD)


def test_parsing_keeps_missing_as_none_and_skips_annual():
    ts = fetch(make_provider(ok_handler))["T2M"]
    assert [d.isoformat() for d in ts.dates] == ["2020-01-01", "2020-02-01", "2020-03-01"]
    assert ts.values == [14.5, 17.0, None]
    assert ts.unit == "C" and ts.unit_from_upstream is True
    assert ts.quality.n_missing == 1 and ts.quality.n_valid == 2
    assert any("annual-summary" in n for n in ts.quality.notes)
    assert any("fill-value" in n for n in ts.quality.notes)
    assert ts.provenance.is_fallback is False and ts.provenance.is_reanalysis is True


def test_unit_falls_back_to_catalog_and_says_so():
    payload = {k: v for k, v in SYNTHETIC_PAYLOAD.items() if k != "parameters"}
    ts = fetch(make_provider(lambda r: httpx.Response(200, json=payload)))["T2M"]
    assert ts.unit == "°C" and ts.unit_from_upstream is False
    assert any("catalog" in n for n in ts.quality.notes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latitude": 91},
        {"longitude": -181},
        {"variables": ["NOT_A_VAR"]},
        {"variables": []},
        {"resolution": "hourly"},
        {"start": "2020-01"},
        {"start": "2021", "end": "2020"},
        {"resolution": "daily", "start": "20200230", "end": "20200301"},
    ],
)
def test_validation_rejects_bad_input_without_calling_nasa(kwargs):
    calls = []
    p = make_provider(lambda r: calls.append(1) or httpx.Response(200, json=SYNTHETIC_PAYLOAD))
    with pytest.raises(ValidationFailed):
        fetch(p, **kwargs)
    assert calls == []


def test_retries_transient_errors_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json=SYNTHETIC_PAYLOAD)

    ts = fetch(make_provider(handler))["T2M"]
    assert len(calls) == 3 and ts.quality.n_valid == 2


def test_gives_up_with_clear_error_and_no_substitute_data():
    calls = []
    p = make_provider(lambda r: calls.append(1) or httpx.Response(503), retries=2)
    with pytest.raises(UpstreamUnavailable):
        fetch(p)
    assert len(calls) == 3


def test_timeout_is_retried_then_reported():
    def handler(request):
        raise httpx.ConnectTimeout("boom")

    with pytest.raises(UpstreamUnavailable):
        fetch(make_provider(handler, retries=1))


def test_upstream_422_is_not_retried():
    calls = []
    p = make_provider(lambda r: calls.append(1) or httpx.Response(422, text="bad params"))
    with pytest.raises(ValidationFailed):
        fetch(p)
    assert len(calls) == 1


def test_unexpected_shape_fails_loudly():
    p = make_provider(lambda r: httpx.Response(200, json={"hello": "world"}))
    with pytest.raises(UpstreamBadResponse):
        fetch(p)


def test_missing_variable_in_response_is_reported():
    payload = {**SYNTHETIC_PAYLOAD, "properties": {"parameter": {}}}
    with pytest.raises(NoDataAvailable):
        fetch(make_provider(lambda r: httpx.Response(200, json=payload)))


def test_second_identical_request_is_served_from_cache(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=SYNTHETIC_PAYLOAD)

    p = make_provider(handler, tmp_path=tmp_path)
    first = fetch(p)["T2M"]
    second = fetch(p)["T2M"]
    assert len(calls) == 1
    assert first.provenance.cached is False and second.provenance.cached is True
    assert first.values == second.values


def test_rate_limiter_delays_when_budget_exhausted():
    async def go():
        lim = SlidingWindowRateLimiter(2, window_seconds=0.3)
        t0 = time.monotonic()
        for _ in range(3):
            await lim.acquire()
        return time.monotonic() - t0

    assert run(go()) >= 0.28
