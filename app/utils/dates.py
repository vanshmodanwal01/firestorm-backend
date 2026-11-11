"""Year-range validation and conversion to POWER's date formats."""
from datetime import date, timedelta

from app.core.errors import ValidationFailed

EARLIEST_YEAR = 1981  # POWER meteorology starts in 1981 (solar in 1984)
# POWER daily data lag real time by roughly a week; stay conservatively behind.
DAILY_LATENCY_DAYS = 10


def validate_year_range(start_year: int, end_year: int, today: date | None = None) -> None:
    today = today or date.today()
    if start_year < EARLIEST_YEAR:
        raise ValidationFailed(f"start_year must be {EARLIEST_YEAR} or later", {"start_year": start_year})
    if end_year > today.year:
        raise ValidationFailed(f"end_year cannot be after {today.year}", {"end_year": end_year})
    if start_year > end_year:
        raise ValidationFailed("start_year must not be after end_year",
                               {"start_year": start_year, "end_year": end_year})


def provider_dates(start_year: int, end_year: int, resolution: str,
                   today: date | None = None) -> tuple[str, str]:
    """Return (start, end) strings in the format POWER expects."""
    today = today or date.today()
    validate_year_range(start_year, end_year, today)
    if resolution == "monthly":
        return str(start_year), str(end_year)
    end = min(date(end_year, 12, 31), today - timedelta(days=DAILY_LATENCY_DAYS))
    start = date(start_year, 1, 1)
    if start > end:
        raise ValidationFailed("start_year is too recent for daily data")
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
