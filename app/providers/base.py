"""Provider interface so new NASA datasets can be added without touching routes."""
from abc import ABC, abstractmethod

from app.schemas.timeseries import TimeSeries


class DataProvider(ABC):
    name: str

    @abstractmethod
    async def fetch_point(
        self,
        variables: list[str],
        latitude: float,
        longitude: float,
        start: str,
        end: str,
        resolution: str = "monthly",
    ) -> dict[str, TimeSeries]:
        """Return one TimeSeries per requested variable.

        Implementations must never fabricate data: missing observations are
        returned as None, and failures raise AppError subclasses.
        """
