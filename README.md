# Earth System Trend Detective: Backend

NASA Space Apps Challenge 2026, "Be An Earth System Trend Detective!" (backend only).

It fetches real **NASA POWER** data, removes the seasonal cycle, tests for a trend, and explains the result in plain language.
Meteorological variables are **MERRA-2 reanalysis** (model-assimilated), not direct instrument readings. The API says so in every response.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows PowerShell; on macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env
uvicorn app.main:app --reload
```

Swagger docs: http://127.0.0.1:8000/docs

Run the tests (offline, no NASA access needed): `python -m pytest -q`

## Endpoints (`/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/variables`, `/variables/{code}` | Variables with plain-language definitions |
| GET | `/observations` | Time series for one variable and location |
| POST | `/trends/analyze` | Full trend analysis (chart-ready series, statistics, insight) |

Every success response is `{"data": ..., "meta": {"request_id": "...", "cached": false}}`.
Every error is `{"error": {"code": "...", "message": "...", "details": {}, "request_id": "..."}}`.

| Status | Code | Meaning |
|---|---|---|
| 404 | `not_found` / `no_data` | Unknown variable / NASA had no data |
| 422 | `validation_error` | Bad input |
| 429 | `upstream_rate_limited` | NASA is throttling us |
| 502 | `upstream_bad_response` | NASA answered in an unexpected shape |
| 503 | `upstream_unavailable` | NASA unreachable after retries (no substitute data is ever generated) |

### Example: analyze a trend

```bash
curl -X POST http://127.0.0.1:8000/api/v1/trends/analyze \
  -H "Content-Type: application/json" \
  -d '{"variable":"T2M","latitude":25.43,"longitude":81.84,"start_year":1985,"end_year":2025}'
```

Key fields of `data`: `trend.classification` (`increasing`, `decreasing`, `no_significant_trend`, `inconclusive`,
`insufficient_data`), `trend.sen_slope` (rate per decade with confidence interval), `trend.mann_kendall.p_value`,
`trend.change` (total change, effect size), `series` (parallel arrays for charts; `null` = missing),
`diagnostics` (data quality and every cleaning action), and `insight` (plain-language summary).

### Example: raw observations

`GET /observations?variable=T2M&latitude=25.43&longitude=81.84&start_year=2000&end_year=2020`

## How trends are tested

1. Monthly values -> subtract each calendar month's average (removes the seasonal cycle) -> annual mean anomalies (a year needs at least 9 valid months).
2. **Mann-Kendall** test with the **Hamed-Rao autocorrelation correction** (floored at 1.0 so the correction can only make the test stricter). At least 10 valid years are required.
3. **Sen's slope** for the rate, with a confidence interval that uses the corrected variance.
4. **OLS** with Newey-West robust errors as a cross-check. **Seasonal Mann-Kendall** on raw monthly values as supporting evidence.
5. A trend is called increasing/decreasing only if the corrected Mann-Kendall p-value is below `alpha` **and** Mann-Kendall, Sen and OLS agree on direction. Otherwise it is `no_significant_trend` or `inconclusive`.
6. Statistical significance is kept separate from practical importance (`effect_size`: change relative to year-to-year variation).

Nothing is imputed. Missing values stay `null`, outliers are flagged but kept, and every data-changing step is listed in `diagnostics.notes`.

## Deploy on Railway

Add the Postgres plugin, set `DATABASE_URL` from it (`postgres://` URLs are converted automatically) and `CORS_ORIGINS` to your frontend's domain.
The `Procfile` supplies the start command. The response cache lives in the database because Railway's disk is ephemeral.
