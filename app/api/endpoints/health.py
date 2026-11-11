from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
async def health() -> dict:
    """Liveness only. It does not call NASA, so it stays fast and never counts against rate limits."""
    return {"status": "ok", "service": "earth-system-trend-detective-api", "version": "0.2.0"}
