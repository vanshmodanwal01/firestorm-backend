from fastapi import APIRouter, Depends, Request

from app.api.dependencies import request_id
from app.core.errors import NotFound
from app.providers.variables import VARIABLES, get_variable
from app.schemas.common import ERROR_RESPONSES, Envelope, Meta
from app.schemas.variables import VariableOut
from app.services.trend_analysis_service import variable_out

router = APIRouter(tags=["variables"])


@router.get(
    "/variables", response_model=Envelope[list[VariableOut]], responses=ERROR_RESPONSES,
    summary="Environmental variables available for analysis",
)
async def list_variables(request: Request, rid: str = Depends(request_id)):
    """Each variable has a plain-language definition suitable for students and the public."""
    data = [variable_out(v) for v in VARIABLES.values()]
    return {"data": data, "meta": Meta(request_id=rid)}


@router.get(
    "/variables/{code}", response_model=Envelope[VariableOut], responses=ERROR_RESPONSES,
    summary="One variable's definition and data-source details",
)
async def get_one_variable(code: str, rid: str = Depends(request_id)):
    info = get_variable(code)
    if info is None:
        raise NotFound(f"unknown variable '{code}'", {"supported": sorted(VARIABLES)})
    return {"data": variable_out(info), "meta": Meta(request_id=rid)}
