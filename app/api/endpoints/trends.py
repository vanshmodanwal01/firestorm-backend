from fastapi import APIRouter, Depends

from app.api.dependencies import get_trend_service, request_id
from app.schemas.common import ERROR_RESPONSES, Envelope, Meta
from app.schemas.trends import TrendAnalysisData, TrendRequest
from app.services.trend_analysis_service import TrendAnalysisService

router = APIRouter(tags=["trends"])


@router.post(
    "/trends/analyze", response_model=Envelope[TrendAnalysisData], responses=ERROR_RESPONSES,
    summary="Detect and test a trend for one variable at one location",
)
async def analyze_trend(
    req: TrendRequest,
    service: TrendAnalysisService = Depends(get_trend_service),
    rid: str = Depends(request_id),
):
    """Answers: what changed, how much, over what period, and whether it is statistically significant.

    Returns chart-ready series, the trend statistics (Mann-Kendall with autocorrelation correction,
    Sen's slope with confidence interval, robust OLS), data-quality diagnostics and a plain-language
    insight generated only from those results.
    """
    data = await service.analyze(req)
    return {"data": data, "meta": Meta(request_id=rid, cached=data.provenance.cached)}
