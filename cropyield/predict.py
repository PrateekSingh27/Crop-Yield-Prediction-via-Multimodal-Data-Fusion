"""Inference service: load the trained bundle and predict + explain one season.

    from cropyield.predict import YieldPredictor
    YieldPredictor().predict("Wheat", {...raw inputs...}, district="Karnal")
"""
from __future__ import annotations

from functools import cached_property

import joblib
import pandas as pd

from .config import FEATURE_INFO, MODEL_BUNDLE_PATH, modality_of
from .explain import contributions
from .features import build_features


class YieldPredictor:
    def __init__(self, bundle_path=MODEL_BUNDLE_PATH):
        if not bundle_path.exists():
            raise FileNotFoundError("Model bundle missing - run: python -m cropyield.train")
        self.bundle = joblib.load(bundle_path)

    @cached_property
    def model(self):
        return self.bundle["model"]

    @property
    def model_name(self) -> str:
        return self.bundle["model_name"]

    def default_inputs(self, crop: str, district: str | None = None) -> dict[str, float]:
        key = f"{district}|{crop}"
        if district and key in self.bundle["presets"]:
            return dict(self.bundle["presets"][key]["inputs"])
        return dict(self.bundle["crop_defaults"][crop])

    def _range_warnings(self, inputs: dict[str, float]) -> list[str]:
        warnings = []
        for name, value in inputs.items():
            r = self.bundle["ranges"].get(name)
            if r and not (r["min"] <= value <= r["max"]):
                warnings.append(f"{FEATURE_INFO[name]['label']} = {value:g} is outside the training range "
                                f"({r['min']:.2f}-{r['max']:.2f}); the estimate is an extrapolation.")
        return warnings

    def predict(self, crop: str, inputs: dict[str, float], district: str | None = None) -> dict:
        full_inputs = {**self.default_inputs(crop, district), **inputs}
        row = build_features(pd.DataFrame([{**full_inputs, "crop": crop}]))
        X = row[self.bundle["features"]]

        prediction = float(self.model.predict(X)[0])
        contrib, base = contributions(self.model, X)
        values = contrib.iloc[0]

        items = []
        for feature, value in values.items():
            items.append({"feature": feature, "label": FEATURE_INFO[feature]["label"],
                          "unit": FEATURE_INFO[feature]["unit"], "modality": modality_of(feature),
                          "input_value": float(X.iloc[0][feature]), "contribution": float(value)})
        items.sort(key=lambda d: abs(d["contribution"]), reverse=True)

        by_modality: dict[str, float] = {}
        for item in items:
            by_modality[item["modality"]] = by_modality.get(item["modality"], 0.0) + item["contribution"]

        interval = self.bundle["intervals"][crop]
        result = {
            "crop": crop, "district": district, "model": self.model_name,
            "predicted_yield_t_ha": round(prediction, 3),
            "interval_80": [round(prediction + interval["lower"], 3), round(prediction + interval["upper"], 3)],
            "base_value": round(float(base[0]), 3),
            "contributions": items,
            "modality_contributions": {k: round(v, 4) for k, v in by_modality.items()},
            "warnings": self._range_warnings(inputs),
        }
        preset = self.bundle["presets"].get(f"{district}|{crop}") if district else None
        if preset:
            result["district_mean_yield"] = round(preset["mean_yield"], 3)
            result["vs_district_mean_pct"] = round(100 * (prediction / preset["mean_yield"] - 1), 1)
        return result
