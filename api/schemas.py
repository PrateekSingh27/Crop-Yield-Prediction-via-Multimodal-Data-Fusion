"""Request / response models with input validation (ranges match FEATURE_INFO)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SeasonInputs(BaseModel):
    """Raw inputs for one district-crop season. Any field left out is filled
    from the district (or crop) typical value."""
    # Satellite (NDVI phase means)
    ndvi_early: Optional[float] = Field(None, ge=0.05, le=0.9, description="Early-season NDVI")
    ndvi_mid: Optional[float] = Field(None, ge=0.05, le=0.95, description="Mid-season NDVI")
    ndvi_peak: Optional[float] = Field(None, ge=0.1, le=0.95, description="Peak-season NDVI")
    ndvi_late: Optional[float] = Field(None, ge=0.05, le=0.9, description="Late-season NDVI")
    # Weather (season aggregates)
    rainfall_mm: Optional[float] = Field(None, ge=0, le=2500)
    rainfall_cv: Optional[float] = Field(None, ge=0, le=4)
    flowering_rainfall_mm: Optional[float] = Field(None, ge=0, le=900)
    dry_spell_days: Optional[float] = Field(None, ge=0, le=150)
    tmax_mean_c: Optional[float] = Field(None, ge=10, le=45)
    tmin_mean_c: Optional[float] = Field(None, ge=-2, le=32)
    solar_radiation_mj: Optional[float] = Field(None, ge=5, le=32)
    rh_mean_pct: Optional[float] = Field(None, ge=5, le=100)
    gdd: Optional[float] = Field(None, ge=0, le=4000)
    heat_stress_days: Optional[float] = Field(None, ge=0, le=40)
    # Soil
    soil_ph: Optional[float] = Field(None, ge=4.0, le=9.5)
    soil_n_kg_ha: Optional[float] = Field(None, ge=50, le=500)
    soil_oc_pct: Optional[float] = Field(None, ge=0.1, le=2.0)
    sand_pct: Optional[float] = Field(None, ge=0, le=100)
    silt_pct: Optional[float] = Field(None, ge=0, le=100)
    clay_pct: Optional[float] = Field(None, ge=0, le=100)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.tmax_mean_c is not None and self.tmin_mean_c is not None and self.tmin_mean_c >= self.tmax_mean_c:
            raise ValueError("tmin_mean_c must be lower than tmax_mean_c")
        texture = [self.sand_pct, self.silt_pct, self.clay_pct]
        if all(v is not None for v in texture) and abs(sum(texture) - 100) > 3:
            raise ValueError("sand + silt + clay must add up to ~100%")
        return self


class PredictRequest(BaseModel):
    crop: Literal["Wheat", "Rice"]
    district: Optional[str] = Field(None, description="Optional; used for defaults and comparison")
    inputs: SeasonInputs = Field(default_factory=SeasonInputs)

    model_config = {"json_schema_extra": {"example": {
        "crop": "Wheat", "district": "Karnal",
        "inputs": {"ndvi_peak": 0.78, "heat_stress_days": 4, "soil_n_kg_ha": 220}}}}
