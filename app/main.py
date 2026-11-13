"""FastAPI application entry point.

Run locally:   uvicorn app.main:app --reload
Railway start: uvicorn app.main:app --host 0.0.0.0 --port $PORT
Interactive docs (Swagger): /docs      OpenAPI JSON: /openapi.json
"""
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.dependencies import get_power_provider
from app.api.endpoints.frontend_bridge import router as bridge_router
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
        title="TerraTrend & Earth System Trend Detective API",
        version="0.2.0",
        description=(
            "Unified Full-Stack Platform for NASA Space Apps 2026. Integrates TerraTrend Frontend, "
            "NASA POWER API, and Trend-Detective statistical engine (Mann-Kendall, Hamed-Rao, Sen's slope)."
        ),
        lifespan=lifespan,
    )
    # Permissive CORS for seamless local testing and multi-port frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
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

    # 1. API Routers
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(bridge_router, prefix="/api")
    app.include_router(bridge_router, prefix="/api/v1")

    # 2. Serve TerraTrend Frontend static assets & pages
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    if os.path.exists(frontend_dir):
        # Mount assets directories
        for folder in ["css", "js"]:
            sub_path = os.path.join(frontend_dir, folder)
            if os.path.exists(sub_path):
                app.mount(f"/{folder}", StaticFiles(directory=sub_path), name=folder)

        @app.get("/", include_in_schema=False)
        async def root_index():
            index_path = os.path.join(frontend_dir, "index.html")
            return FileResponse(index_path)

        @app.get("/{page_name}.html", include_in_schema=False)
        async def serve_page(page_name: str):
            target = os.path.join(frontend_dir, f"{page_name}.html")
            if os.path.exists(target):
                return FileResponse(target)
            return JSONResponse(status_code=404, content={"detail": f"Page '{page_name}.html' not found"})

    return app


app = create_app()
