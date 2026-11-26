"""Central configuration: paths, study area, crop calendars and feature groups.

Everything that other modules need to agree on lives here, so the simulator,
the real-data (NASA POWER) path, the training pipeline and the API all use
exactly the same definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NASA_DIR = RAW_DIR / "nasa_power"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

DATASET_PATH = DATA_DIR / "crop_yield_multimodal.csv"
DAILY_WEATHER_PATH = DATA_DIR / "daily_weather.csv"
NDVI_SERIES_PATH = DATA_DIR / "ndvi_composites.csv"
MODEL_BUNDLE_PATH = MODEL_DIR / "model_bundle.joblib"

RANDOM_SEED = 42
YEARS = list(range(2014, 2025))  # 11 crop seasons -> 11 LOYO folds

# --------------------------------------------------------------------------
# Study area: 32 districts across 11 Indian states (approximate centroids)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class District:
    name: str
    state: str
    lat: float
    lon: float
    crops: tuple[str, ...]


DISTRICTS: list[District] = [
    District("Ludhiana", "Punjab", 30.90, 75.85, ("Wheat", "Rice")),
    District("Patiala", "Punjab", 30.34, 76.39, ("Wheat", "Rice")),
    District("Amritsar", "Punjab", 31.63, 74.87, ("Wheat", "Rice")),
    District("Karnal", "Haryana", 29.69, 76.99, ("Wheat", "Rice")),
    District("Hisar", "Haryana", 29.15, 75.72, ("Wheat",)),
    District("Faridabad", "Haryana", 28.41, 77.32, ("Wheat",)),
    District("Jaipur", "Rajasthan", 26.91, 75.79, ("Wheat",)),
    District("Kota", "Rajasthan", 25.21, 75.86, ("Wheat",)),
    District("Ajmer", "Rajasthan", 26.45, 74.64, ("Wheat",)),
    District("Meerut", "Uttar Pradesh", 28.98, 77.71, ("Wheat",)),
    District("Agra", "Uttar Pradesh", 27.18, 78.01, ("Wheat",)),
    District("Kanpur", "Uttar Pradesh", 26.45, 80.33, ("Wheat", "Rice")),
    District("Indore", "Madhya Pradesh", 22.72, 75.86, ("Wheat",)),
    District("Bhopal", "Madhya Pradesh", 23.26, 77.41, ("Wheat",)),
    District("Ujjain", "Madhya Pradesh", 23.18, 75.78, ("Wheat",)),
    District("Nashik", "Maharashtra", 20.00, 73.79, ("Wheat",)),
    District("Pune", "Maharashtra", 18.52, 73.86, ("Rice",)),
    District("Nagpur", "Maharashtra", 21.15, 79.09, ("Rice",)),
    District("Guntur", "Andhra Pradesh", 16.31, 80.44, ("Rice",)),
    District("Kurnool", "Andhra Pradesh", 15.83, 78.04, ("Rice",)),
    District("Anantapur", "Andhra Pradesh", 14.68, 77.60, ("Rice",)),
    District("Mysuru", "Karnataka", 12.30, 76.64, ("Rice",)),
    District("Dharwad", "Karnataka", 15.46, 75.01, ("Wheat",)),
    District("Belagavi", "Karnataka", 15.85, 74.50, ("Rice",)),
    District("Coimbatore", "Tamil Nadu", 11.02, 76.96, ("Rice",)),
    District("Madurai", "Tamil Nadu", 9.93, 78.12, ("Rice",)),
    District("Salem", "Tamil Nadu", 11.66, 78.15, ("Rice",)),
    District("Patna", "Bihar", 25.59, 85.14, ("Wheat", "Rice")),
    District("Gaya", "Bihar", 24.79, 85.00, ("Wheat", "Rice")),
    District("Purnia", "Bihar", 25.78, 87.47, ("Wheat", "Rice")),
    District("Raipur", "Chhattisgarh", 21.25, 81.63, ("Rice",)),
    District("Bilaspur", "Chhattisgarh", 22.08, 82.15, ("Rice",)),
]
DISTRICT_BY_NAME = {d.name: d for d in DISTRICTS}

CROPS = ("Wheat", "Rice")

# --------------------------------------------------------------------------
# Crop calendars (simplified, typical North/Central India practice)
#   Wheat = Rabi (sown Nov, harvested Apr) -> season labelled by harvest year
#   Rice  = Kharif (transplanted late Jun, harvested late Oct)
# Phase windows are "days after sowing" (DAS).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CropCalendar:
    sow_month: int
    sow_day: int
    season_days: int
    sow_in_previous_year: bool
    flowering_das: tuple[int, int]
    grain_fill_das: tuple[int, int]
    gdd_base_c: float
    gdd_cap_c: float
    heat_threshold_c: float          # Tmax above this during sensitive window = stress day
    water_need_mm: float             # seasonal crop water requirement (approx.)


CALENDARS: dict[str, CropCalendar] = {
    "Wheat": CropCalendar(11, 10, 150, True, (95, 115), (105, 140), 5.0, 30.0, 33.0, 420.0),
    "Rice": CropCalendar(6, 20, 125, False, (70, 92), (85, 115), 10.0, 35.0, 35.0, 1100.0),
}

# Phase windows for NDVI phase means (days after sowing)
NDVI_PHASES = {
    "ndvi_early": (0, 40),
    "ndvi_mid": (40, 80),
    "ndvi_peak": (80, 110),
    "ndvi_late": (110, 150),
}

# --------------------------------------------------------------------------
# Feature groups (= modalities). Raw columns are stored in the CSV; derived
# columns are created in features.py.
# --------------------------------------------------------------------------
TARGET = "yield_tonnes_ha"
ID_COLUMNS = ["district", "state", "crop", "year", "lat", "lon", "weather_source"]

SATELLITE_RAW = ["ndvi_early", "ndvi_mid", "ndvi_peak", "ndvi_late"]
WEATHER_RAW = [
    "rainfall_mm", "rainfall_cv", "flowering_rainfall_mm", "dry_spell_days",
    "tmax_mean_c", "tmin_mean_c", "solar_radiation_mj", "rh_mean_pct",
    "gdd", "heat_stress_days",
]
SOIL_RAW = ["soil_ph", "soil_n_kg_ha", "soil_oc_pct", "sand_pct", "silt_pct", "clay_pct"]

SATELLITE_DERIVED = ["ndvi_growth_slope", "ndvi_senescence", "ndvi_auc"]
WEATHER_DERIVED = ["temp_range_c"]
INTERACTION = ["rainfall_x_soil_n"]
CONTEXT = ["crop_is_rice"]

MODALITIES: dict[str, list[str]] = {
    "Satellite": SATELLITE_RAW + SATELLITE_DERIVED,
    "Weather": WEATHER_RAW + WEATHER_DERIVED,
    "Soil": SOIL_RAW,
}
ALL_FEATURES: list[str] = (
    MODALITIES["Satellite"] + MODALITIES["Weather"] + MODALITIES["Soil"] + INTERACTION + CONTEXT
)

# Human-readable metadata used by the API / frontend
FEATURE_INFO: dict[str, dict] = {
    "ndvi_early": {"label": "Early-season NDVI", "unit": "index", "min": 0.05, "max": 0.9},
    "ndvi_mid": {"label": "Mid-season NDVI", "unit": "index", "min": 0.05, "max": 0.95},
    "ndvi_peak": {"label": "Peak-season NDVI", "unit": "index", "min": 0.1, "max": 0.95},
    "ndvi_late": {"label": "Late-season NDVI", "unit": "index", "min": 0.05, "max": 0.9},
    "ndvi_growth_slope": {"label": "NDVI green-up slope", "unit": "index/phase"},
    "ndvi_senescence": {"label": "NDVI senescence drop", "unit": "index x100"},
    "ndvi_auc": {"label": "Seasonal NDVI integral", "unit": "index-phases"},
    "rainfall_mm": {"label": "Seasonal rainfall", "unit": "mm", "min": 0, "max": 2500},
    "rainfall_cv": {"label": "Weekly rainfall variability (CV)", "unit": "ratio", "min": 0, "max": 4},
    "flowering_rainfall_mm": {"label": "Rainfall at flowering", "unit": "mm", "min": 0, "max": 900},
    "dry_spell_days": {"label": "Longest dry spell", "unit": "days", "min": 0, "max": 150},
    "tmax_mean_c": {"label": "Mean max temperature", "unit": "°C", "min": 10, "max": 45},
    "tmin_mean_c": {"label": "Mean min temperature", "unit": "°C", "min": -2, "max": 32},
    "solar_radiation_mj": {"label": "Solar radiation", "unit": "MJ/m²/day", "min": 5, "max": 32},
    "rh_mean_pct": {"label": "Mean relative humidity", "unit": "%", "min": 5, "max": 100},
    "gdd": {"label": "Growing degree days", "unit": "°C·day", "min": 0, "max": 4000},
    "heat_stress_days": {"label": "Heat-stress days (sensitive stage)", "unit": "days", "min": 0, "max": 40},
    "temp_range_c": {"label": "Diurnal temperature range", "unit": "°C"},
    "soil_ph": {"label": "Soil pH", "unit": "pH", "min": 4.0, "max": 9.5},
    "soil_n_kg_ha": {"label": "Available nitrogen", "unit": "kg/ha", "min": 50, "max": 500},
    "soil_oc_pct": {"label": "Organic carbon", "unit": "%", "min": 0.1, "max": 2.0},
    "sand_pct": {"label": "Sand", "unit": "%", "min": 0, "max": 100},
    "silt_pct": {"label": "Silt", "unit": "%", "min": 0, "max": 100},
    "clay_pct": {"label": "Clay", "unit": "%", "min": 0, "max": 100},
    "rainfall_x_soil_n": {"label": "Rainfall × soil N interaction", "unit": "mm·kg/ha"},
    "crop_is_rice": {"label": "Crop = Rice", "unit": "0/1"},
}


def modality_of(feature: str) -> str:
    """Return which modality a feature belongs to (used for SHAP roll-ups)."""
    for name, cols in MODALITIES.items():
        if feature in cols:
            return name
    if feature in INTERACTION:
        return "Cross-modal"
    return "Context"
