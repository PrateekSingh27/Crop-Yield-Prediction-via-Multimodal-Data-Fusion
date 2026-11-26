"""Turn DAILY data into SEASON-level features.

This is the "temporal alignment" step from the PPT: daily weather (synthetic or
real NASA POWER) and NDVI composites are aggregated into crop-stage windows
defined by the crop calendar. The exact same functions are used for synthetic
and real data, so the model never sees two different definitions of a feature.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import CALENDARS, NDVI_PHASES


def sowing_date(crop: str, season_year: int) -> date:
    """Sowing date for a crop season. Wheat season 2020 = sown Nov 2019, harvested Apr 2020."""
    cal = CALENDARS[crop]
    year = season_year - 1 if cal.sow_in_previous_year else season_year
    return date(year, cal.sow_month, cal.sow_day)


def season_window(crop: str, season_year: int) -> tuple[date, date]:
    start = sowing_date(crop, season_year)
    end = start + timedelta(days=CALENDARS[crop].season_days - 1)
    return start, end


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def weather_features(daily: pd.DataFrame, crop: str, season_year: int) -> dict[str, float]:
    """Aggregate daily weather for one crop season.

    `daily` needs columns: date, rainfall_mm, tmax_c, tmin_c, solar_radiation_mj, rh_pct.
    Days are assumed to be the crop season only (use `slice_season` first).
    """
    cal = CALENDARS[crop]
    d = daily.sort_values("date").reset_index(drop=True)
    das = np.arange(len(d))  # days after sowing
    rain = d["rainfall_mm"].to_numpy(dtype=float)
    tmax = d["tmax_c"].to_numpy(dtype=float)
    tmin = d["tmin_c"].to_numpy(dtype=float)
    tmean = np.clip((tmax + tmin) / 2.0, cal.gdd_base_c, cal.gdd_cap_c)

    weekly = np.add.reduceat(rain, np.arange(0, len(rain), 7))
    weekly_mean = weekly.mean()
    rainfall_cv = float(weekly.std() / weekly_mean) if weekly_mean > 0 else 0.0

    flowering = (das >= cal.flowering_das[0]) & (das <= cal.flowering_das[1])
    sensitive = (das >= cal.flowering_das[0]) & (das <= cal.grain_fill_das[1])

    return {
        "rainfall_mm": float(np.nansum(rain)),
        "rainfall_cv": rainfall_cv,
        "flowering_rainfall_mm": float(np.nansum(rain[flowering])),
        "dry_spell_days": float(_longest_run(rain < 1.0)),
        "tmax_mean_c": float(np.nanmean(tmax)),
        "tmin_mean_c": float(np.nanmean(tmin)),
        "solar_radiation_mj": float(np.nanmean(d["solar_radiation_mj"])),
        "rh_mean_pct": float(np.nanmean(d["rh_pct"])),
        "gdd": float(np.nansum(tmean - cal.gdd_base_c)),
        "heat_stress_days": float(np.sum(tmax[sensitive] > cal.heat_threshold_c)),
    }


def slice_season(daily: pd.DataFrame, crop: str, season_year: int) -> pd.DataFrame:
    start, end = season_window(crop, season_year)
    dates = pd.to_datetime(daily["date"]).dt.date
    return daily.loc[(dates >= start) & (dates <= end)]


def ndvi_phase_means(das: np.ndarray, ndvi: np.ndarray) -> dict[str, float]:
    """Average NDVI composites falling in each phenological phase window."""
    out = {}
    for name, (lo, hi) in NDVI_PHASES.items():
        mask = (das >= lo) & (das < hi)
        out[name] = float(np.nanmean(ndvi[mask])) if mask.any() else float("nan")
    return out
