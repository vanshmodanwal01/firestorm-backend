"""Plain-language summaries generated ONLY from computed results.

Every number in the text is read from the analysis result. Nothing is
invented, and no cause is claimed: the tool says what changed and whether the
change is statistically significant, never why it happened.
"""
from __future__ import annotations

from app.providers.variables import VariableInfo
from app.schemas.timeseries import Provenance
from app.schemas.trends import Diagnostics, TrendResult


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 0.001:
        return f"{x:.2e}"
    return f"{x:.3g}"


def _where(lat: float, lon: float) -> str:
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"


def generate_insight(info: VariableInfo, unit: str, lat: float, lon: float,
                     trend: TrendResult, diag: Diagnostics, provenance: Provenance) -> str:
    unit = {"C": "°C"}.get(unit, unit)  # display only; the data field keeps NASA's string
    what = info.name[0].lower() + info.name[1:]
    where = _where(lat, lon)
    parts: list[str] = []

    if trend.classification == "insufficient_data" or trend.sen_slope is None or trend.change is None:
        reason = trend.reasons[0] if trend.reasons else "The record is too short or has too many gaps."
        parts.append(
            f"There is not enough valid data to judge whether {what} at {where} is trending. {reason}"
        )
    else:
        sen, ch = trend.sen_slope, trend.change
        period = f"{ch.period_start_year}–{ch.period_end_year}"
        ci = f"{_fmt(sen.ci_low_per_decade)} to {_fmt(sen.ci_high_per_decade)}"
        level = f"{round(sen.confidence_level * 100)}%"
        rate = f"{_fmt(abs(sen.slope_per_decade))} {unit} per decade"

        if trend.classification in ("increasing", "decreasing"):
            word = "upward" if trend.classification == "increasing" else "downward"
            parts.append(
                f"Over {period}, {what} at {where} shows a statistically significant {word} trend "
                f"of about {rate} ({level} confidence interval: {ci})."
            )
            total = f"roughly {_fmt(abs(ch.total_change))} {unit}"
            if ch.pct_change is not None:
                total += f" ({_fmt(abs(ch.pct_change))}% of the period average)"
            parts.append(f"Across the whole period that adds up to {total}.")
            if ch.effect_size_label:
                parts.append(
                    f"This change is {ch.effect_size_label} compared with normal year-to-year variation."
                )
                if ch.effect_size_label == "small":
                    parts.append(
                        "A statistically significant change is not automatically an important one."
                    )
        elif trend.classification == "inconclusive":
            parts.append(
                f"Over {period}, the statistical tests for {what} at {where} did not agree on the "
                f"direction of change, so no trend is claimed. The best estimate is {_fmt(sen.slope_per_decade)} "
                f"{unit} per decade ({level} confidence interval: {ci}); treat it with caution."
            )
        else:
            parts.append(
                f"Over {period}, no statistically significant trend was found in {what} at {where}. "
                f"The best estimate is a change of {_fmt(sen.slope_per_decade)} {unit} per decade "
                f"({level} confidence interval: {ci}), but the data are too variable, or the record too "
                f"short, to tell that apart from random year-to-year variation."
            )
            parts.append(
                "A trend that is not significant is still a finding, and a longer record can sometimes "
                "make a real trend clear."
            )

        if trend.mann_kendall and trend.mann_kendall.variance_inflation_factor > 1.2:
            parts.append(
                "Because neighbouring years were similar to each other, the significance test was made "
                "stricter to avoid false alarms."
            )
        if trend.seasonal_mann_kendall and not trend.seasonal_mann_kendall.agrees_with_primary:
            parts.append(
                "A separate month-by-month test reached a different conclusion, which is another reason "
                "for caution."
            )

    if diag.missing_fraction > 0.05:
        parts.append(f"About {round(diag.missing_fraction * 100)}% of the months in this record had no data.")
    if provenance.is_reanalysis:
        parts.append(
            "These values come from the MERRA-2 model reanalysis used by NASA POWER, which blends "
            "satellite and other observations in a model; they are not a single instrument at this spot."
        )
    else:
        parts.append(
            "These values are satellite-based estimates from NASA POWER, not ground measurements at this spot."
        )
    return " ".join(parts)
