"""Environment-based application settings (no secrets are hardcoded)."""
from functools import lru_cache

from pydantic import field_validator
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

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        # Railway (and Heroku-style hosts) hand out postgres:// or postgresql://
        # URLs; SQLAlchemy needs an explicit driver.
        if v.startswith("postgres://"):
            v = "postgresql+psycopg2://" + v[len("postgres://"):]
        elif v.startswith("postgresql://"):
            v = "postgresql+psycopg2://" + v[len("postgresql://"):]
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
