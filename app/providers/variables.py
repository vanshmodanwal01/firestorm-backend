"""Catalog of NASA POWER variables supported by this backend.

Parameter codes come from NASA POWER's own documentation examples
(T2M, T2M_MAX, PRECTOTCORR, RH2M, WS10M, ALLSKY_SFC_SW_DWN). Units here are
only a FALLBACK: when NASA's response states a unit we always use that, and
the TimeSeries records which one was used (`unit_from_upstream`).

Plain-language definitions are for the student/public audience described in
the challenge's "Potential Considerations".
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class VariableInfo:
    code: str
    name: str
    definition: str
    fallback_unit: str
    category: str
    is_reanalysis: bool
    dataset_note: str
    coverage_start_year: int
    # False for interval-scale quantities such as degrees Celsius, where
    # "percent change" is meaningless because 0 °C is not a true zero.
    ratio_scale: bool = True


_MET_NOTE = "NASA POWER meteorological parameter derived from MERRA-2 model reanalysis (not a direct instrument reading)."
_SOLAR_NOTE = "NASA POWER solar parameter derived from satellite-based radiation products (CERES and, before 2000, GEWEX SRB)."

VARIABLES: dict[str, VariableInfo] = {
    v.code: v
    for v in [
        VariableInfo(
            "T2M", "Air temperature at 2 m",
            "Average air temperature about 2 metres above the ground, roughly the height of a weather station thermometer.",
            "°C", "temperature", True, _MET_NOTE, 1981, ratio_scale=False,
        ),
        VariableInfo(
            "T2M_MAX", "Daily maximum air temperature at 2 m",
            "The hottest air temperature reached during the day (or the average of daily maxima, for monthly data).",
            "°C", "temperature", True, _MET_NOTE, 1981, ratio_scale=False,
        ),
        VariableInfo(
            "T2M_MIN", "Daily minimum air temperature at 2 m",
            "The coolest air temperature reached during the day (or the average of daily minima, for monthly data).",
            "°C", "temperature", True, _MET_NOTE, 1981, ratio_scale=False,
        ),
        VariableInfo(
            "PRECTOTCORR", "Precipitation (bias-corrected)",
            "Rain and snow that fell, expressed as a depth of water per day. Rainfall is very variable, so trends are harder to confirm.",
            "mm/day", "precipitation", True, _MET_NOTE, 1981,
        ),
        VariableInfo(
            "RH2M", "Relative humidity at 2 m",
            "How much water vapour the air holds compared with the most it could hold at that temperature.",
            "%", "humidity", True, _MET_NOTE, 1981,
        ),
        VariableInfo(
            "WS10M", "Wind speed at 10 m",
            "Average wind speed about 10 metres above the ground.",
            "m/s", "wind", True, _MET_NOTE, 1981,
        ),
        VariableInfo(
            "ALLSKY_SFC_SW_DWN", "Sunlight reaching the surface (all-sky)",
            "Solar energy arriving at the ground, including the effect of clouds.",
            "MJ/m^2/day", "solar", False, _SOLAR_NOTE, 1984,  # confirmed by live probe 2026-11-10
        ),
    ]
}


def get_variable(code: str) -> VariableInfo | None:
    return VARIABLES.get(code.upper())
