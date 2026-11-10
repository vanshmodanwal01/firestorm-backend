"""Environment-based application settings (no secrets are hardcoded)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./nasa_trends.db"
    log_level: str = "INFO"

    power_base_url: str = "https://power.larc.nasa.gov"
    power_max_requests_per_minute: int = 30
    power_timeout_seconds: float = 60.0
    power_max_retries: int = 3
    cache_ttl_hours: int = 168

    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
