"""Response envelope and error shapes shared by every endpoint."""
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Meta(BaseModel):
    request_id: str
    cached: bool | None = Field(default=None, description="True if NASA data came from our cache")


class Envelope(BaseModel, Generic[T]):
    data: T
    meta: Meta


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Not found / no data"},
    422: {"model": ErrorResponse, "description": "Validation error"},
    429: {"model": ErrorResponse, "description": "NASA rate limit"},
    502: {"model": ErrorResponse, "description": "Unexpected NASA response"},
    503: {"model": ErrorResponse, "description": "NASA unavailable"},
}
