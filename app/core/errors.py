"""Domain exceptions shared by providers and services.

The API layer maps these to structured HTTP error responses.
"""


class AppError(Exception):
    """Base class. `code` is a stable machine-readable identifier."""

    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationFailed(AppError):
    code = "validation_error"
    http_status = 422


class UpstreamUnavailable(AppError):
    """NASA service could not be reached or kept failing after retries."""

    code = "upstream_unavailable"
    http_status = 503


class UpstreamRateLimited(AppError):
    code = "upstream_rate_limited"
    http_status = 429


class UpstreamBadResponse(AppError):
    """NASA answered, but not in a shape we can safely interpret."""

    code = "upstream_bad_response"
    http_status = 502


class NotFound(AppError):
    code = "not_found"
    http_status = 404


class NoDataAvailable(AppError):
    code = "no_data"
    http_status = 404
