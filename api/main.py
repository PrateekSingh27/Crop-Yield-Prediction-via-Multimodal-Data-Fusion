"""FastAPI backend for the Crop Yield Multimodal Fusion project.

Run:   python -m uvicorn api.main:app --port 8000
Docs:  http://localhost:8000/docs     (interactive Swagger UI)
UI:    http://localhost:8000          (built React dashboard, served from api/static)
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

from cropyield.config import (DATASET_PATH, DISTRICTS, FEATURE_INFO, FIGURE_DIR, MODEL_BUNDLE_PATH,
                              NDVI_SERIES_PATH, REPORT_DIR, SATELLITE_RAW, SOIL_RAW, TARGET, WEATHER_RAW)
from cropyield.predict import YieldPredictor

from .schemas import PredictRequest

log = logging.getLogger("uvicorn.error")
STATIC_DIR = Path(__file__).parent / "static"
SWAGGER_DIR = Path(__file__).parent / "swagger"  # bundled so /docs works offline
STATE: dict = {}


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (NaN -> null, numpy -> python)."""
    clean = df.replace([np.inf, -np.inf], np.nan).astype(object).where(pd.notna(df), None)
    return json.loads(clean.to_json(orient="records"))


def load_state() -> None:
    if not MODEL_BUNDLE_PATH.exists():
        log.warning("No trained model found - training now (about 1 minute)...")
        from cropyield.train import run
        run()
    STATE["predictor"] = YieldPredictor()
    STATE["data"] = pd.read_csv(DATASET_PATH)
    STATE["ndvi"] = pd.read_csv(NDVI_SERIES_PATH) if NDVI_SERIES_PATH.exists() else pd.DataFrame()
    STATE["reports"] = {p.stem: pd.read_csv(p) for p in REPORT_DIR.glob("*.csv")}
    STATE["summary"] = json.loads((REPORT_DIR / "training_summary.json").read_text())
    log.info("Loaded model %s and %d rows", STATE["predictor"].model_name, len(STATE["data"]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield


app = FastAPI(title="Crop Yield Prediction via Multimodal Data Fusion",
              description="Satellite NDVI + weather + soil -> early-fusion gradient boosting with LOYO validation "
                          "and SHAP explanations.", version="2.0.0", lifespan=lifespan, docs_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------
@app.get("/docs", include_in_schema=False)
def swagger_docs():
    """Swagger UI served from local files (no CDN), so it works without internet."""
    return get_swagger_ui_html(openapi_url=app.openapi_url, title=f"{app.title} - API docs",
                               swagger_js_url="/swagger/swagger-ui-bundle.js",
                               swagger_css_url="/swagger/swagger-ui.css",
                               swagger_favicon_url="/favicon.ico")


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "model": STATE["predictor"].model_name, "rows": len(STATE["data"])}


@app.get("/api/meta", tags=["system"])
def meta():
    """Everything the UI needs to build forms: inputs, ranges, defaults, districts."""
    bundle = STATE["predictor"].bundle
    groups = {"Satellite": SATELLITE_RAW, "Weather": WEATHER_RAW, "Soil": SOIL_RAW}
    inputs = []
    for modality, cols in groups.items():
        for c in cols:
            info, r = FEATURE_INFO[c], bundle["ranges"][c]
            inputs.append({"name": c, "label": info["label"], "unit": info["unit"], "modality": modality,
                           "min": info["min"], "max": info["max"], "p01": r["p01"], "p99": r["p99"]})
    return {
        "model": STATE["predictor"].model_name,
        "summary": STATE["summary"],
        "inputs": inputs,
        "crop_defaults": bundle["crop_defaults"],
        "presets": bundle["presets"],
        "districts": [{"district": d.name, "state": d.state, "lat": d.lat, "lon": d.lon, "crops": list(d.crops)}
                      for d in DISTRICTS],
    }


@app.get("/api/overview", tags=["data"])
def overview():
    df = STATE["data"]
    by_state = df.groupby(["state", "crop"], as_index=False)[TARGET].mean().round(3)
    by_year = df.groupby(["year", "crop"], as_index=False)[TARGET].mean().round(3)
    points = (df.groupby(["district", "state", "crop", "lat", "lon"], as_index=False)
              .agg(mean_yield=(TARGET, "mean"), mean_ndvi_peak=("ndvi_peak", "mean")).round(3))
    return {
        "kpis": {"rows": len(df), "districts": int(df.district.nunique()), "states": int(df.state.nunique()),
                 "years": [int(df.year.min()), int(df.year.max())], "crops": sorted(df.crop.unique()),
                 "mean_yield": round(float(df[TARGET].mean()), 3), "weather_source": str(df.weather_source.iloc[0])},
        "by_state": records(by_state), "by_year": records(by_year), "map_points": records(points),
    }


@app.get("/api/ndvi-strips", tags=["data"])
def ndvi_strips():
    """Mean NDVI per crop x season x 16-day composite, plus mean yield per season."""
    nd, df = STATE["ndvi"], STATE["data"]
    if nd.empty:
        return {"crops": {}}
    out = {}
    for crop, g in nd.groupby("crop"):
        grid = g.groupby(["year", "days_after_sowing"], as_index=False)["ndvi"].mean().round(3)
        yields = df[df.crop == crop].groupby("year")[TARGET].mean().round(3)
        out[crop] = {"das": sorted(grid.days_after_sowing.unique().tolist()),
                     "years": [{"year": int(y), "mean_yield": float(yields[y]),
                                "ndvi": grid[grid.year == y].sort_values("days_after_sowing")["ndvi"].tolist()}
                               for y in sorted(grid.year.unique())]}
    return {"crops": out}


@app.get("/api/dataset", tags=["data"])
def dataset(crop: Optional[str] = None, state: Optional[str] = None, district: Optional[str] = None,
            limit: int = Query(500, le=5000), offset: int = 0):
    df = STATE["data"]
    if crop:
        df = df[df.crop == crop]
    if state:
        df = df[df.state == state]
    if district:
        df = df[df.district == district]
    return {"total": len(df), "rows": records(df.iloc[offset:offset + limit].round(3))}


@app.get("/api/district/{name}", tags=["data"])
def district_detail(name: str, crop: str = "Wheat"):
    df = STATE["data"]
    sub = df[(df.district == name) & (df.crop == crop)].sort_values("year")
    if sub.empty:
        raise HTTPException(404, f"No {crop} data for district '{name}'")
    history = sub[["year", TARGET, "ndvi_peak", "rainfall_mm", "heat_stress_days", "soil_n_kg_ha"]].round(3)
    curves = []
    nd = STATE["ndvi"]
    if not nd.empty:
        s = nd[(nd.district == name) & (nd.crop == crop)]
        pivot = s.pivot_table(index="days_after_sowing", columns="year", values="ndvi")
        best_year = int(sub.loc[sub[TARGET].idxmax(), "year"])
        worst_year = int(sub.loc[sub[TARGET].idxmin(), "year"])
        for das, row in pivot.iterrows():
            curves.append({"das": int(das), "mean": round(float(row.mean()), 3),
                           "min": round(float(row.min()), 3), "max": round(float(row.max()), 3),
                           "best": round(float(row[best_year]), 3), "worst": round(float(row[worst_year]), 3)})
    else:
        best_year = worst_year = None
    return {"district": name, "crop": crop, "state": sub.state.iloc[0], "history": records(history),
            "ndvi_curve": curves, "best_year": best_year, "worst_year": worst_year,
            "mean_yield": round(float(sub[TARGET].mean()), 3)}


@app.get("/api/metrics", tags=["model"])
def model_metrics():
    r = STATE["reports"]
    pick = {k: records(r[k].round(4)) for k in ("model_comparison", "modality_ablation", "per_year_metrics",
                                                 "per_crop_metrics", "per_state_metrics", "spatial_validation",
                                                 "feature_importance_shap", "modality_contribution") if k in r}
    return {**pick, "summary": STATE["summary"]}


@app.get("/api/predictions", tags=["model"])
def loyo_predictions(model: Optional[str] = None):
    preds = STATE["reports"]["loyo_predictions"]
    model = model or STATE["predictor"].model_name
    sub = preds[preds.model == model]
    if sub.empty:
        raise HTTPException(404, f"Unknown model '{model}'")
    return {"model": model, "rows": records(sub.drop(columns="model").round(3))}


@app.post("/api/predict", tags=["model"])
def predict(request: PredictRequest):
    predictor: YieldPredictor = STATE["predictor"]
    if request.district and f"{request.district}|{request.crop}" not in predictor.bundle["presets"]:
        raise HTTPException(422, f"{request.district} has no {request.crop} data; choose another district or crop")
    inputs = {k: v for k, v in request.inputs.model_dump().items() if v is not None}
    return predictor.predict(request.crop, inputs, request.district)


# --------------------------------------------------------------------------
# Static files: report figures + the built React app (must be mounted last)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=FIGURE_DIR), name="figures")
app.mount("/swagger", StaticFiles(directory=SWAGGER_DIR), name="swagger")
if (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
else:
    @app.get("/", include_in_schema=False)
    def no_ui():
        return {"message": "API running. Frontend not built - see README (cd web; npm install; npm run build).",
                "docs": "/docs"}
