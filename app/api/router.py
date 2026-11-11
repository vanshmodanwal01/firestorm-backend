from fastapi import APIRouter

from app.api.endpoints import health, observations, trends, variables

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(variables.router)
api_router.include_router(observations.router)
api_router.include_router(trends.router)
