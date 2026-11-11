"""FastAPI application entry point.

Run locally:   uvicorn app.main:app --reload
Railway start: uvicorn app.main:app --host 0.0.0.0 --port $PORT
Interactive docs (Swagger): /docs      OpenAPI JSON: /openapi.json
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dependencies import get_power_provider
from app.api.router import api_router
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if get_power_provider.cache_info().currsize:
        await get_power_provider().aclose()


def _error(status: int, code: str, message: str, details: dict, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details,
                           "request_id": getattr(request.state, "request_id", None)}},
    )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Earth System Trend Detective API",
        version="0.2.0",
        description=(
            "Backend for NASA Space Apps 2026 'Be An Earth System Trend Detective!'. Retrieves real "
            "NASA POWER data, removes the seasonal cycle, tests for trends (Mann-Kendall with "
            "autocorrelation correction, Sen's slope, robust OLS) and explains the result in plain language. "
            "Meteorological values are MERRA-2 reanalysis, not direct instrument readings."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        if exc.http_status >= 500:
            logger.warning("%s: %s", exc.code, exc.message)
        return _error(exc.http_status, exc.code, exc.message, exc.details, request)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        details = {"errors": jsonable_encoder(exc.errors(), custom_encoder={Exception: str})}
        return _error(422, "validation_error", "request validation failed", details, request)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.exception("unhandled error")
        return _error(500, "internal_error", "an unexpected error occurred", {}, request)

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        return {"name": app.title, "docs": "/docs", "health": "/api/v1/health"}

    return app


app = create_app()
