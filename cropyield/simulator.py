"""Agronomic simulator that produces the multimodal training dataset.

WHY A SIMULATOR?
Official district yields, Sentinel-2/MODIS NDVI and Soil Health Card data need
accounts, GIS processing and days of cleaning. For a reproducible academic
project we simulate them with crop-science response curves, and keep a switch
to use REAL NASA POWER weather (see nasa_power.py).

WHAT IS SIMULATED (per district x crop x season):
1. Daily weather   - seasonal temperature cycle, monsoon rainfall, solar, humidity.
                     Year-to-year monsoon strength follows approximate IMD
                     all-India departures; 2022 includes the March heatwave
                     that hit North-Indian wheat.
2. Soil            - state soil profile + fixed district offset + small yearly
                     Soil Health Card sampling noise.
3. Crop growth     - yield = potential x management x f(water) x f(heat) x f(N)
                     x f(pH) x f(OC) x f(texture) x f(flowering rain) x f(radiation)
                     with non-linear (threshold / saturating) response curves.
4. NDVI            - 16-day MODIS-style composites from a double-logistic canopy
                     curve whose height follows crop vigour and whose decline
                     starts early under heat/drought; monsoon clouds add noise.

"Management" (irrigation, seed variety, practices) is NOT given to the model,
exactly like reality. Satellite NDVI partly "sees" it through canopy vigour,
which is why fusing satellite with weather and soil helps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CALENDARS, DISTRICTS, YEARS, District
from .season import ndvi_phase_means, season_window, weather_features


# --------------------------------------------------------------------------
# State-level climate, management and soil profiles (approximate, rounded)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StateProfile:
    kharif_rain_mm: float      # Jun-Oct normal rainfall
    rabi_rain_mm: float        # Nov-Apr normal rainfall
    tmean_c: float             # annual mean temperature
    t_amplitude_c: float       # seasonal amplitude
    irrigation: dict           # crop -> irrigated fraction
    management: dict           # crop -> management / technology index (hidden)
    ph: float
    n_kg_ha: float
    oc_pct: float
    sand_pct: float
    clay_pct: float
    ne_monsoon: bool = False   # receives North-East (Oct-Dec) monsoon


STATES: dict[str, StateProfile] = {
    "Punjab": StateProfile(520, 110, 24.0, 9.5, {"Wheat": .98, "Rice": .99}, {"Wheat": 1.00, "Rice": 1.00}, 8.0, 210, .50, 48, 18),
    "Haryana": StateProfile(480, 80, 24.8, 9.5, {"Wheat": .92, "Rice": .95}, {"Wheat": .93, "Rice": .86}, 8.0, 185, .45, 52, 17),
    "Rajasthan": StateProfile(520, 35, 25.5, 8.5, {"Wheat": .75, "Rice": .5}, {"Wheat": .78, "Rice": .6}, 8.2, 150, .30, 62, 14),
    "Uttar Pradesh": StateProfile(800, 55, 25.8, 8.5, {"Wheat": .85, "Rice": .70}, {"Wheat": .80, "Rice": .72}, 7.8, 190, .45, 38, 22),
    "Madhya Pradesh": StateProfile(950, 45, 25.0, 7.0, {"Wheat": .70, "Rice": .4}, {"Wheat": .74, "Rice": .55}, 7.5, 200, .55, 25, 45),
    "Maharashtra": StateProfile(800, 40, 26.0, 4.5, {"Wheat": .45, "Rice": .35}, {"Wheat": .55, "Rice": .62}, 7.7, 190, .55, 22, 48),
    "Andhra Pradesh": StateProfile(600, 200, 28.3, 3.8, {"Wheat": .5, "Rice": .90}, {"Wheat": .5, "Rice": .88}, 7.8, 200, .50, 35, 35, True),
    "Karnataka": StateProfile(650, 120, 24.8, 3.2, {"Wheat": .40, "Rice": .65}, {"Wheat": .48, "Rice": .76}, 6.8, 230, .60, 40, 30),
    "Tamil Nadu": StateProfile(380, 400, 28.0, 3.2, {"Wheat": .5, "Rice": .85}, {"Wheat": .5, "Rice": .86}, 7.2, 210, .55, 45, 28, True),
    "Bihar": StateProfile(1050, 45, 25.8, 7.5, {"Wheat": .70, "Rice": .55}, {"Wheat": .64, "Rice": .60}, 7.2, 230, .55, 30, 25),
    "Chhattisgarh": StateProfile(1150, 55, 26.5, 6.0, {"Wheat": .4, "Rice": .35}, {"Wheat": .5, "Rice": .57}, 6.3, 240, .65, 35, 30),
}

POTENTIAL_YIELD = {"Wheat": 6.4, "Rice": 5.6}  # t/ha under ideal conditions

# Approximate IMD all-India south-west monsoon departure from normal (%).
MONSOON_DEPARTURE = {2014: -12, 2015: -14, 2016: -3, 2017: -5, 2018: -9, 2019: 10,
                     2020: 9, 2021: -1, 2022: 6, 2023: -6, 2024: 8}
# Spring (Feb-Apr) temperature anomaly for the wheat season, deg C. 2022 = March heatwave.
SPRING_HEAT_ANOMALY = {2014: -0.3, 2015: -0.6, 2016: 0.6, 2017: 0.4, 2018: 0.5, 2019: -0.4,
                       2020: -0.7, 2021: 0.8, 2022: 2.8, 2023: 0.7, 2024: 0.1}

# Normalised seasonal temperature profile by month (-1 = coldest, +1 = hottest), mid-month anchors
_TEMP_PROFILE = np.array([-1.0, -0.7, -0.1, 0.55, 1.0, 0.95, 0.55, 0.40, 0.35, 0.10, -0.45, -0.85])
_MONSOON_WETNESS = np.array([0, 0, 0, 0, 0.05, 0.5, 1.0, 1.0, 0.75, 0.3, 0.05, 0])
_KHARIF_RAIN_SHAPE = np.array([0, 0, 0, 0, 0.1, 0.55, 1.0, 0.95, 0.6, 0.25, 0.05, 0])
_RABI_RAIN_SHAPE = np.array([0.6, 0.7, 0.5, 0.3, 0.1, 0, 0, 0, 0, 0.1, 0.3, 0.3])
_RABI_RAIN_SHAPE_NE = np.array([0.25, 0.1, 0.05, 0.1, 0.1, 0, 0, 0, 0, 0.6, 1.0, 0.7])


def _monthly_interp(table: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Smoothly interpolate 12 mid-month values to daily values (circular)."""
    x = np.arange(12) * 30.44 + 15.2
    xp = np.concatenate([x - 365.25, x, x + 365.25])
    fp = np.concatenate([table, table, table])
    return np.interp(np.asarray(dates.day_of_year), xp, fp)


# --------------------------------------------------------------------------
# Hidden, fixed district characteristics
# --------------------------------------------------------------------------
@dataclass
class DistrictTraits:
    management: dict
    rain_factor: float
    soil: dict


def make_district_traits(rng: np.random.Generator) -> dict[str, DistrictTraits]:
    traits = {}
    for d in DISTRICTS:
        sp = STATES[d.state]
        mgmt = {c: float(np.clip(sp.management[c] * rng.lognormal(0, 0.06), 0.35, 1.08)) for c in ("Wheat", "Rice")}
        sand = float(np.clip(sp.sand_pct + rng.normal(0, 6), 8, 80))
        clay = float(np.clip(sp.clay_pct + rng.normal(0, 5), 6, 60))
        soil = {
            "ph": float(np.clip(sp.ph + rng.normal(0, 0.3), 5.0, 9.0)),
            "n": float(sp.n_kg_ha + rng.normal(0, 25) + 60 * (np.mean(list(mgmt.values())) - 0.75)),
            "oc": float(np.clip(sp.oc_pct + rng.normal(0, 0.08), 0.15, 1.3)),
            "sand": sand,
            "clay": clay,
        }
        traits[d.name] = DistrictTraits(mgmt, float(rng.lognormal(0, 0.12)), soil)
    return traits


# --------------------------------------------------------------------------
# 1. Daily weather
# --------------------------------------------------------------------------
def synthetic_daily_weather(district: District, crop: str, season_year: int,
                            traits: DistrictTraits, year_noise: dict,
                            rng: np.random.Generator) -> pd.DataFrame:
    sp = STATES[district.state]
    start, end = season_window(crop, season_year)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    profile = _monthly_interp(_TEMP_PROFILE, dates)
    wetness = _monthly_interp(_MONSOON_WETNESS, dates)

    # --- rainfall ---------------------------------------------------------
    if crop == "Rice":
        shape = _monthly_interp(_KHARIF_RAIN_SHAPE, dates)
        departure = MONSOON_DEPARTURE[season_year] / 100.0
        year_factor = (1 + 1.6 * departure) * year_noise["kharif_region"][district.state]
        normal = sp.kharif_rain_mm
        p_max = 0.68
    else:
        shape = _monthly_interp(_RABI_RAIN_SHAPE_NE if sp.ne_monsoon else _RABI_RAIN_SHAPE, dates)
        year_factor = year_noise["rabi"] * year_noise["rabi_region"][district.state]
        normal = sp.rabi_rain_mm
        p_max = 0.14
    target_total = normal * traits.rain_factor * year_factor * rng.lognormal(0, 0.12)
    wet_prob = np.clip(p_max * shape / max(shape.max(), 1e-6), 0.01, 0.85)
    wet = rng.random(n) < wet_prob
    amounts = np.where(wet, rng.gamma(0.75, 1.0, n) * (0.3 + shape), 0.0)
    rain = amounts * (target_total / amounts.sum()) if amounts.sum() > 0 else amounts

    # --- temperature ------------------------------------------------------
    anomaly = np.zeros(n)
    noise = rng.normal(0, 1.2, n)
    for i in range(1, n):  # AR(1) day-to-day persistence
        anomaly[i] = 0.8 * anomaly[i - 1] + noise[i]
    anomaly += year_noise["temp"][district.state]
    if crop == "Wheat":
        spring = ((dates.month >= 2) & (dates.month <= 4)).astype(float)
        anomaly += spring * (SPRING_HEAT_ANOMALY[season_year] + rng.normal(0, 0.3))
    tmean = sp.tmean_c + sp.t_amplitude_c * profile + anomaly - 1.0 * wet
    dtr = 13.0 - 5.0 * wetness - 3.0 * wet  # diurnal range shrinks when cloudy/wet
    tmax = tmean + dtr / 2
    tmin = tmean - dtr / 2

    # --- radiation & humidity --------------------------------------------
    winter_sun_loss = 0.18 * (district.lat - 18) * np.clip(-profile, 0, 1)  # low winter sun up north
    solar = (19.5 + 4.5 * profile - 3.0 * wetness - 4.0 * wet - 0.08 * rain
             - winter_sun_loss + rng.normal(0, 1.0, n))
    rh = 45 + 30 * wetness + 15 * wet - 8 * np.clip(profile, 0, 1) + rng.normal(0, 4, n)

    return pd.DataFrame({
        "date": dates.date,
        "rainfall_mm": rain,
        "tmax_c": tmax,
        "tmin_c": tmin,
        "solar_radiation_mj": np.clip(solar, 6, 30),
        "rh_pct": np.clip(rh, 12, 98),
    })


def make_year_noise(rng: np.random.Generator) -> dict[int, dict]:
    states = list(STATES)
    out = {}
    for year in YEARS:
        out[year] = {
            "kharif_region": {s: float(rng.lognormal(0, 0.15)) for s in states},
            "rabi": float(rng.lognormal(0, 0.35)),
            "rabi_region": {s: float(rng.lognormal(0, 0.2)) for s in states},
            "temp": {s: float(rng.normal(0, 0.5)) for s in states},
            "shock": {s: float(rng.lognormal(0, 0.05)) for s in states},  # pests/disease etc.
        }
    return out


# --------------------------------------------------------------------------
# 2. Soil (Soil Health Card style yearly samples)
# --------------------------------------------------------------------------
def sample_soil(traits: DistrictTraits, rng: np.random.Generator) -> dict[str, float]:
    s = traits.soil
    sand = float(np.clip(s["sand"] + rng.normal(0, 2), 5, 85))
    clay = float(np.clip(s["clay"] + rng.normal(0, 2), 5, 65))
    silt = max(5.0, 100 - sand - clay)
    total = sand + clay + silt
    return {
        "soil_ph": float(np.clip(s["ph"] + rng.normal(0, 0.12), 4.5, 9.3)),
        "soil_n_kg_ha": float(np.clip(s["n"] + rng.normal(0, 15), 60, 450)),
        "soil_oc_pct": float(np.clip(s["oc"] + rng.normal(0, 0.04), 0.1, 1.5)),
        "sand_pct": sand * 100 / total,
        "silt_pct": silt * 100 / total,
        "clay_pct": clay * 100 / total,
    }


# --------------------------------------------------------------------------
# 3. Crop response curves
# --------------------------------------------------------------------------
def crop_response(crop: str, w: dict, soil: dict, irrigation: float) -> dict[str, float]:
    """Multiplicative stress factors (1 = no stress). Deliberately non-linear."""
    cal = CALENDARS[crop]
    need = cal.water_need_mm
    rain = w["rainfall_mm"]
    supply = rain + irrigation * max(0.0, need - rain) * 0.95
    holding = 1 + 0.004 * (soil["clay_pct"] - 25) - 0.003 * (soil["sand_pct"] - 40)
    ratio = supply * holding / need

    f_water = 1 - 0.85 * max(0.0, 1 - ratio) ** 1.3
    if ratio > 1.5:  # waterlogging / floods
        f_water -= 0.12 * min(1.0, ratio - 1.5)
    if crop == "Rice":
        f_water -= 0.004 * max(0.0, w["dry_spell_days"] - 12) * (1 - irrigation)
    f_water = float(np.clip(f_water, 0.15, 1.0))

    f_heat = float(np.exp(-(0.022 if crop == "Wheat" else 0.03) * w["heat_stress_days"]))

    fr = w["flowering_rainfall_mm"]
    if crop == "Rice":
        f_flower = 1 - 0.10 * (1 - irrigation) * max(0.0, 40 - fr) / 40
    else:  # rain + wind at flowering -> lodging in wheat
        f_flower = 1 - 0.08 * min(1.0, max(0.0, fr - 50) / 80)

    f_n = 1 - 0.75 * np.exp(-0.012 * soil["soil_n_kg_ha"])          # saturating (Mitscherlich)
    f_ph = float(np.clip(1 - 0.07 * (soil["soil_ph"] - 6.8) ** 2, 0.7, 1.0))  # optimum ~6.8
    f_oc = 0.88 + 0.2 * min(soil["soil_oc_pct"], 1.0)
    f_tex = 1 + 0.004 * (soil["clay_pct"] - 25) if crop == "Rice" else 1 - 0.003 * abs(soil["sand_pct"] - 40)
    f_rad = float(np.clip(0.88 + 0.02 * (w["solar_radiation_mj"] - 14), 0.85, 1.05))

    return {"water": f_water, "heat": f_heat, "flower": f_flower, "n": float(f_n),
            "ph": f_ph, "oc": f_oc, "texture": f_tex, "radiation": f_rad}


# --------------------------------------------------------------------------
# 4. NDVI composites
# --------------------------------------------------------------------------
def ndvi_composites(crop: str, vigor: float, early_senescence_days: float,
                    cloudiness: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cal = CALENDARS[crop]
    das = np.arange(0, cal.season_days, 16).astype(float)  # 16-day composites
    t_up = 38.0 if crop == "Wheat" else 32.0
    t_down = cal.season_days - 18 - early_senescence_days
    canopy = 1 / (1 + np.exp(-(das - t_up) / 8)) - 1 / (1 + np.exp(-(das - t_down) / 7))
    ndvi = 0.17 + 0.66 * vigor * canopy
    cloudy = rng.random(len(das)) < cloudiness
    ndvi -= cloudy * rng.uniform(0.04, 0.22, len(das))        # residual cloud contamination
    ndvi += rng.normal(0, 0.02, len(das))                       # sensor / mixed-pixel noise
    return das, np.clip(ndvi, 0.05, 0.95)


# --------------------------------------------------------------------------
# Assemble one district-season record
# --------------------------------------------------------------------------
def simulate_record(district: District, crop: str, year: int, daily: pd.DataFrame,
                    traits: DistrictTraits, year_noise: dict, rng: np.random.Generator,
                    weather_source: str) -> tuple[dict, pd.DataFrame]:
    sp = STATES[district.state]
    irrigation = sp.irrigation[crop]
    mgmt = traits.management[crop]

    w = weather_features(daily, crop, year)
    soil = sample_soil(traits, rng)
    f = crop_response(crop, w, soil, irrigation)
    trend = 1 + 0.012 * (year - 2014)  # technology / variety improvement

    # Canopy vigour seen by the satellite (management + water + nitrogen)
    vigor = mgmt ** 0.7 * (0.55 + 0.45 * f["water"]) * f["n"] * trend ** 0.5 * rng.lognormal(0, 0.04)
    vigor = float(np.clip(vigor, 0.2, 1.08))
    early_sen = 1.6 * w["heat_stress_days"] + 25 * (1 - f["water"])
    cloudiness = 0.40 if crop == "Rice" else 0.07
    das, ndvi = ndvi_composites(crop, vigor, early_sen, cloudiness, rng)

    stress = np.prod(list(f.values()))
    y = (POTENTIAL_YIELD[crop] * mgmt * stress * trend
         * year_noise["shock"][district.state] * rng.lognormal(0, 0.05))

    record = {
        "district": district.name, "state": district.state, "crop": crop, "year": year,
        "lat": district.lat, "lon": district.lon, "weather_source": weather_source,
        **ndvi_phase_means(das, ndvi), **w, **soil,
        "yield_tonnes_ha": round(float(np.clip(y, 0.6, 7.5)), 3),
    }
    series = pd.DataFrame({"district": district.name, "crop": crop, "year": year,
                           "days_after_sowing": das.astype(int), "ndvi": ndvi})
    return record, series
