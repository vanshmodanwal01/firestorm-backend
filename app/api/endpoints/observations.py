from typing import Literal

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_provider, request_id
from app.core.errors import ValidationFailed
from app.providers.base import DataProvider
from app.providers.variables import VARIABLES, get_variable
from app.schemas.common import ERROR_RESPONSES, Envelope, Meta
from app.schemas.timeseries import TimeSeries
from app.utils.dates import provider_dates

router = APIRouter(tags=["observations"])


@router.get(
    "/observations", response_model=Envelope[TimeSeries], responses=ERROR_RESPONSES,
    summary="Historical time series for one variable at one location",
)
async def get_observations(
    variable: str = Query(..., examples=["T2M"]),
    latitude: float = Query(..., ge=-90, le=90, examples=[25.43]),
    longitude: float = Query(..., ge=-180, le=180, examples=[81.84]),
    start_year: int = Query(..., ge=1981, examples=[2000]),
    end_year: int = Query(..., ge=1981, examples=[2020]),
    resolution: Literal["monthly", "daily"] = "monthly",
    provider: DataProvider = Depends(get_provider),
    rid: str = Depends(request_id),
):
    """Real NASA POWER values. Missing observations are returned as null and are never filled in."""
    info = get_variable(variable)
    if info is None:
        raise ValidationFailed(f"unsupported variable '{variable}'", {"supported": sorted(VARIABLES)})
    start, end = provider_dates(start_year, end_year, resolution)
    result = await provider.fetch_point([info.code], latitude, longitude, start, end, resolution)
    series = result[info.code]
    return {"data": series, "meta": Meta(request_id=rid, cached=series.provenance.cached)}
