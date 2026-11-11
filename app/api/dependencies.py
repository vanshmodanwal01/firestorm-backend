"""FastAPI dependencies. Tests replace `get_provider` with a test double."""
from functools import lru_cache

from fastapi import Depends, Request

from app.cache.db_cache import DbCache
from app.core.config import get_settings
from app.providers.base import DataProvider
from app.providers.power import PowerProvider
from app.services.trend_analysis_service import TrendAnalysisService


@lru_cache
def get_power_provider() -> PowerProvider:
    settings = get_settings()
    cache = DbCache(settings.database_url, settings.cache_ttl_hours)
    return PowerProvider(settings, cache=cache)


def get_provider() -> DataProvider:
    return get_power_provider()


def get_trend_service(provider: DataProvider = Depends(get_provider)) -> TrendAnalysisService:
    return TrendAnalysisService(provider)


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")
