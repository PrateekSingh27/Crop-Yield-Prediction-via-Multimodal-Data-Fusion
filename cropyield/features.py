"""Feature engineering shared by training and the prediction API.

Raw modality columns come from the dataset; this module adds the derived
phenology, weather and cross-modal interaction features listed in the PPT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALL_FEATURES, CONTEXT, INTERACTION, MODALITIES, TARGET


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with every model feature present."""
    x = df.copy()

    # --- Satellite-derived phenology -------------------------------------
    x["ndvi_growth_slope"] = (x["ndvi_peak"] - x["ndvi_early"]) / 2.0     # green-up rate per phase
    x["ndvi_senescence"] = (x["ndvi_peak"] - x["ndvi_late"]) * 100.0      # canopy decline after peak
    phases = x[["ndvi_early", "ndvi_mid", "ndvi_peak", "ndvi_late"]].to_numpy()
    x["ndvi_auc"] = np.trapezoid(phases, axis=1) if hasattr(np, "trapezoid") else np.trapz(phases, axis=1)

    # --- Weather-derived ---------------------------------------------------
    x["temp_range_c"] = x["tmax_mean_c"] - x["tmin_mean_c"]

    # --- Cross-modal interaction (weather x soil) -------------------------
    x["rainfall_x_soil_n"] = x["rainfall_mm"] * x["soil_n_kg_ha"]

    # --- Context -----------------------------------------------------------
    x["crop_is_rice"] = (x["crop"] == "Rice").astype(int)
    return x


def feature_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    return df[columns or ALL_FEATURES]


def modality_columns(names: list[str]) -> list[str]:
    """Columns for a combination of modalities, always including crop context.

    The rainfall x soil-N interaction is only included when both Weather and
    Soil are present, so single-modality ablations stay "pure".
    """
    cols: list[str] = []
    for name in names:
        cols += MODALITIES[name]
    if {"Weather", "Soil"} <= set(names):
        cols += INTERACTION
    return cols + CONTEXT


def target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET]
