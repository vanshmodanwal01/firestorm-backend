"""API contract tests.

The NASA provider is replaced by a test double that returns SYNTHETIC data
(labeled as such), so these tests check our API's behaviour offline.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_provider
from app.core.errors import NoDataAvailable, UpstreamUnavailable
from app.main import create_app
from app.providers.base import DataProvider
from tests.helpers import synthetic_monthly


class FakeProvider(DataProvider):
    name = "fake"

    def __init__(self):
        self.error = None
        self.calls = []

    async def fetch_point(self, variables, latitude, longitude, start, end, resolution="monthly"):
        self.calls.append((variables, latitude, longitude, start, end, resolution))
        if self.error:
            raise self.error
        return {variables[0]: synthetic_monthly(
            start_year=int(start[:4]), end_year=int(end[:4]), variable=variables[0],
            lat=latitude, lon=longitude,
        )}


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture
def client(fake):
    app = create_app()
    app.dependency_overrides[get_provider] = lambda: fake
    return TestClient(app, raise_server_exceptions=False)


BODY = {"variable": "T2M", "latitude": 25.43, "longitude": 81.84, "start_year": 1985, "end_year": 2024}


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["X-Request-ID"]


def test_variables_list_has_plain_language_definitions(client):
    r = client.get("/api/v1/variables")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"data", "meta"} and body["meta"]["request_id"]
    codes = {v["code"] for v in body["data"]}
    assert {"T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"} <= codes
    t2m = next(v for v in body["data"] if v["code"] == "T2M")
    assert t2m["definition"] and t2m["is_reanalysis"] is True and t2m["percent_change_meaningful"] is False


def test_single_variable_and_unknown_variable_404(client):
    assert client.get("/api/v1/variables/t2m").status_code == 200
    r = client.get("/api/v1/variables/NOPE")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "not_found" and err["request_id"] and "supported" in err["details"]


def test_observations_returns_chart_ready_series(client, fake):
    r = client.get("/api/v1/observations", params={
        "variable": "T2M", "latitude": 25.43, "longitude": 81.84, "start_year": 2000, "end_year": 2002})
    assert r.status_code == 200
    d = r.json()["data"]
    assert len(d["dates"]) == len(d["values"]) == 36
    assert d["provenance"]["is_reanalysis"] is True and d["provenance"]["is_fallback"] is False
    assert fake.calls[0][3:5] == ("2000", "2002")


def test_daily_dates_are_converted_for_nasa(client, fake):
    client.get("/api/v1/observations", params={
        "variable": "T2M", "latitude": 1, "longitude": 1, "start_year": 2010, "end_year": 2012,
        "resolution": "daily"})
    assert fake.calls[0][3] == "20100101" and fake.calls[0][4] == "20121231"


def test_trend_analysis_contract(client):
    r = client.post("/api/v1/trends/analyze", json=BODY)
    assert r.status_code == 200
    body = r.json()
    d = body["data"]
    assert set(body) == {"data", "meta"}
    for key in ("variable", "unit", "location", "series", "trend", "diagnostics", "insight", "provenance"):
        assert key in d
    t = d["trend"]
    assert t["classification"] == "increasing" and t["significant"] is True
    assert 0 < t["mann_kendall"]["p_value"] < 0.05
    assert t["sen_slope"]["ci_low_per_decade"] < t["sen_slope"]["slope_per_decade"] < t["sen_slope"]["ci_high_per_decade"]
    assert isinstance(d["insight"], str) and len(d["insight"]) > 50
    assert len(d["series"]["annual_years"]) == len(d["series"]["trend_line"])


@pytest.mark.parametrize("patch", [
    {"latitude": 123}, {"longitude": -500}, {"start_year": 1900}, {"alpha": 0.9}, {"resolution": "hourly"},
])
def test_bad_input_gives_structured_422(client, patch):
    r = client.post("/api/v1/trends/analyze", json={**BODY, **patch})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error" and err["request_id"]


def test_unsupported_variable_and_bad_range_are_422(client):
    r = client.post("/api/v1/trends/analyze", json={**BODY, "variable": "BOGUS"})
    assert r.status_code == 422 and "supported" in r.json()["error"]["details"]
    r = client.post("/api/v1/trends/analyze", json={**BODY, "start_year": 2020, "end_year": 2010})
    assert r.status_code == 422
    r = client.post("/api/v1/trends/analyze", json={**BODY, "end_year": 2999})
    assert r.status_code == 422


def test_nasa_outage_is_503_with_no_substitute_data(client, fake):
    fake.error = UpstreamUnavailable("NASA POWER unavailable after 4 attempts")
    r = client.post("/api/v1/trends/analyze", json=BODY)
    assert r.status_code == 503
    body = r.json()
    assert "data" not in body and body["error"]["code"] == "upstream_unavailable"


def test_no_data_is_404(client, fake):
    fake.error = NoDataAvailable("NASA POWER returned no data for T2M")
    assert client.post("/api/v1/trends/analyze", json=BODY).status_code == 404


def test_unexpected_bug_is_hidden_behind_generic_500(client, fake):
    fake.error = RuntimeError("secret internal detail")
    r = client.post("/api/v1/trends/analyze", json=BODY)
    assert r.status_code == 500
    assert "secret" not in r.text and r.json()["error"]["code"] == "internal_error"


def test_cors_allows_configured_frontend_origin(client):
    r = client.options("/api/v1/variables", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_openapi_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/v1/health", "/api/v1/variables", "/api/v1/variables/{code}",
              "/api/v1/observations", "/api/v1/trends/analyze"):
        assert p in paths


def test_database_url_is_normalized_for_railway():
    from app.core.config import Settings
    assert Settings(database_url="postgres://u:p@h:5432/db").database_url == "postgresql+psycopg2://u:p@h:5432/db"
    assert Settings(database_url="postgresql://u:p@h/db").database_url.startswith("postgresql+psycopg2://")
    assert Settings(database_url="sqlite:///./x.db").database_url == "sqlite:///./x.db"
