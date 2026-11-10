"""Source-agnostic time-series schemas shared by providers, services and API."""
from datetime import date

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    source: str = Field(description="Data provider, e.g. 'NASA POWER'")
    dataset_note: str = Field(description="Plain statement of what the data physically is")
    is_reanalysis: bool = Field(description="True if model-assimilated (e.g. MERRA-2), not a direct measurement")
    is_fallback: bool = Field(default=False, description="True only for labeled development fallback data")
    cached: bool = False
    request_url: str | None = None


class DataQuality(BaseModel):
    n_total: int
    n_valid: int
    n_missing: int
    missing_fraction: float
    notes: list[str] = Field(default_factory=list)


class TimeSeries(BaseModel):
    variable: str
    unit: str
    unit_from_upstream: bool = Field(description="False means unit came from our catalog, not NASA's response")
    temporal_resolution: str
    latitude: float
    longitude: float
    dates: list[date]
    values: list[float | None] = Field(description="None marks a missing observation; nothing is imputed")
    quality: DataQuality
    provenance: Provenance
