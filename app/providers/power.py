"""NASA POWER provider (https://power.larc.nasa.gov, public, no API key).

Responsibilities: validate input, rate-limit, retry with backoff, cache the raw
response, and convert it into source-agnostic TimeSeries objects.

IMPORTANT: the response parser follows the POWER JSON layout as documented by
NASA's API (values under properties.parameter.<CODE>, fill value and units in
header/parameters). It is strict: if the shape is not what we expect it raises
UpstreamBadResponse instead of guessing. Verify against a real response with
scripts/probe_power.py.
"""
import asyncio
import logging
import re
from datetime import date

import httpx

from app.cache.db_cache import DbCache
from app.core.config import Settings
from app.core.errors import (
    NoDataAvailable,
    UpstreamBadResponse,
    UpstreamRateLimited,
    UpstreamUnavailable,
    ValidationFailed,
)
from app.providers.base import DataProvider
from app.providers.variables import VARIABLES, get_variable
from app.schemas.timeseries import DataQuality, Provenance, TimeSeries
from app.utils.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

DEFAULT_FILL_VALUE = -999.0
MAX_PARAMS_PER_REQUEST = 20  # documented POWER limit for daily/monthly point requests
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class PowerProvider(DataProvider):
    name = "NASA POWER"

    def __init__(
        self,
        settings: Settings,
        cache: DbCache | None = None,
        client: httpx.AsyncClient | None = None,
        limiter: SlidingWindowRateLimiter | None = None,
        backoff_base: float = 1.5,
    ):
        self.settings = settings
        self.cache = cache
        self._client = client
        self._owns_client = client is None
        self.limiter = limiter or SlidingWindowRateLimiter(settings.power_max_requests_per_minute)
        self.backoff_base = backoff_base

    # ------------------------------------------------------------------ public
    async def fetch_point(
        self,
        variables: list[str],
        latitude: float,
        longitude: float,
        start: str,
        end: str,
        resolution: str = "monthly",
    ) -> dict[str, TimeSeries]:
        codes = self._validate(variables, latitude, longitude, start, end, resolution)
        lat, lon = round(latitude, 4), round(longitude, 4)
        params = {
            "parameters": ",".join(codes),
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start,
            "end": end,
            "format": "JSON",
        }
        url = f"{self.settings.power_base_url}/api/temporal/{resolution}/point"
        cache_key = f"power:{resolution}:" + "&".join(f"{k}={params[k]}" for k in sorted(params))

        payload = await self.cache.get(cache_key) if self.cache else None
        cached = payload is not None
        if payload is None:
            payload = await self._request_json(url, params)
            if self.cache:
                await self.cache.set(cache_key, payload)

        request_url = str(httpx.URL(url, params=params))
        return self._parse(payload, codes, lat, lon, resolution, cached, request_url)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    # -------------------------------------------------------------- validation
    @staticmethod
    def _validate(variables, latitude, longitude, start, end, resolution) -> list[str]:
        if resolution not in ("daily", "monthly"):
            raise ValidationFailed("resolution must be 'daily' or 'monthly'", {"resolution": resolution})
        if not -90 <= latitude <= 90:
            raise ValidationFailed("latitude must be between -90 and 90", {"latitude": latitude})
        if not -180 <= longitude <= 180:
            raise ValidationFailed("longitude must be between -180 and 180", {"longitude": longitude})
        if not variables:
            raise ValidationFailed("at least one variable is required")
        if len(variables) > MAX_PARAMS_PER_REQUEST:
            raise ValidationFailed(f"at most {MAX_PARAMS_PER_REQUEST} variables per request")

        codes: list[str] = []
        for v in variables:
            info = get_variable(v)
            if info is None:
                raise ValidationFailed(
                    f"unsupported variable '{v}'", {"supported": sorted(VARIABLES)}
                )
            if info.code not in codes:
                codes.append(info.code)

        pattern = r"^\d{4}$" if resolution == "monthly" else r"^\d{8}$"
        fmt = "YYYY" if resolution == "monthly" else "YYYYMMDD"
        for label, val in (("start", start), ("end", end)):
            if not re.match(pattern, str(val)):
                raise ValidationFailed(f"{label} must be {fmt} for {resolution} data", {label: val})
        if str(start) > str(end):
            raise ValidationFailed("start must not be after end", {"start": start, "end": end})
        if resolution == "daily":
            for label, val in (("start", start), ("end", end)):
                try:
                    date(int(val[:4]), int(val[4:6]), int(val[6:]))
                except ValueError:
                    raise ValidationFailed(f"{label} is not a valid calendar date", {label: val})
        return codes

    # ------------------------------------------------------------------- HTTP
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.power_timeout_seconds)
        return self._client

    async def _request_json(self, url: str, params: dict) -> dict:
        client = await self._get_client()
        attempts = self.settings.power_max_retries + 1
        last_error = "unknown error"

        for attempt in range(1, attempts + 1):
            await self.limiter.acquire()
            try:
                resp = await client.get(url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("POWER request failed (attempt %d/%d): %s", attempt, attempts, last_error)
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError:
                        raise UpstreamBadResponse("NASA POWER returned a non-JSON body")
                if resp.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                    logger.warning("POWER returned %s (attempt %d/%d)", resp.status_code, attempt, attempts)
                    if resp.status_code == 429 and attempt == attempts:
                        raise UpstreamRateLimited("NASA POWER is rate limiting requests; try again shortly")
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit() and attempt < attempts:
                        await asyncio.sleep(min(int(retry_after), 30))
                        continue
                elif resp.status_code in (400, 422):
                    raise ValidationFailed(
                        "NASA POWER rejected the request", {"upstream_status": resp.status_code, "upstream_body": resp.text[:500]}
                    )
                else:
                    raise UpstreamBadResponse(
                        f"unexpected response from NASA POWER (HTTP {resp.status_code})",
                        {"upstream_body": resp.text[:500]},
                    )
            if attempt < attempts:
                await asyncio.sleep(self.backoff_base ** attempt)

        raise UpstreamUnavailable(
            f"NASA POWER unavailable after {attempts} attempts ({last_error})",
            {"hint": "Retry later; no substitute data was generated."},
        )

    # ---------------------------------------------------------------- parsing
    def _parse(self, payload: dict, codes: list[str], lat: float, lon: float,
               resolution: str, cached: bool, request_url: str) -> dict[str, TimeSeries]:
        try:
            parameter_block = payload["properties"]["parameter"]
        except (KeyError, TypeError):
            raise UpstreamBadResponse(
                "NASA POWER response lacks properties.parameter", {"top_level_keys": list(payload)[:10] if isinstance(payload, dict) else None}
            )
        if not isinstance(parameter_block, dict):
            raise UpstreamBadResponse("properties.parameter is not an object")

        header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
        notes_global: list[str] = []
        fill = header.get("fill_value")
        if fill is None:
            fill = DEFAULT_FILL_VALUE
            notes_global.append(f"fill_value absent in response; assumed {DEFAULT_FILL_VALUE}")
        try:
            fill = float(fill)
        except (TypeError, ValueError):
            raise UpstreamBadResponse("header.fill_value is not numeric", {"fill_value": fill})

        units_block = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
        results: dict[str, TimeSeries] = {}

        for code in codes:
            raw = parameter_block.get(code)
            if not isinstance(raw, dict) or not raw:
                raise NoDataAvailable(f"NASA POWER returned no data for {code}", {"variable": code})

            info = VARIABLES[code]
            notes = list(notes_global)
            upstream_unit = units_block.get(code, {}).get("units") if isinstance(units_block.get(code), dict) else None
            unit = upstream_unit or info.fallback_unit
            if not upstream_unit:
                notes.append("unit taken from backend catalog because response did not state one")

            dates: list[date] = []
            values: list[float | None] = []
            n_fill = n_bad = n_annual = 0

            for key in sorted(raw):
                d = self._parse_key(key, resolution)
                if d == "annual":
                    n_annual += 1  # POWER's per-year summary row, not a monthly observation
                    continue
                if d is None:
                    n_bad += 1
                    continue
                val = raw[key]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    values.append(None); dates.append(d); n_bad += 1
                elif float(val) == fill:
                    values.append(None); dates.append(d); n_fill += 1
                else:
                    values.append(float(val)); dates.append(d)

            if n_annual:
                notes.append(f"{n_annual} annual-summary row(s) excluded; they are not monthly observations")
            if n_fill:
                notes.append(f"{n_fill} fill-value observation(s) kept as missing (None), not imputed")
            if n_bad:
                notes.append(f"{n_bad} row(s) had unparseable keys or non-numeric values")

            n_total = len(values)
            n_valid = sum(v is not None for v in values)
            if n_total == 0:
                raise NoDataAvailable(f"no usable observations for {code}", {"variable": code})

            results[code] = TimeSeries(
                variable=code,
                unit=unit,
                unit_from_upstream=bool(upstream_unit),
                temporal_resolution=resolution,
                latitude=lat,
                longitude=lon,
                dates=dates,
                values=values,
                quality=DataQuality(
                    n_total=n_total, n_valid=n_valid, n_missing=n_total - n_valid,
                    missing_fraction=round((n_total - n_valid) / n_total, 6), notes=notes,
                ),
                provenance=Provenance(
                    source=self.name, dataset_note=info.dataset_note,
                    is_reanalysis=info.is_reanalysis, is_fallback=False,
                    cached=cached, request_url=request_url,
                ),
            )
        return results

    @staticmethod
    def _parse_key(key: str, resolution: str):
        """Return a date, the string 'annual', or None if the key is malformed."""
        try:
            if resolution == "monthly":
                if not re.fullmatch(r"\d{6}", key):
                    return None
                year, month = int(key[:4]), int(key[4:])
                if month == 13:
                    return "annual"
                return date(year, month, 1)
            if not re.fullmatch(r"\d{8}", key):
                return None
            return date(int(key[:4]), int(key[4:6]), int(key[6:]))
        except ValueError:
            return None
