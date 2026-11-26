"""End-to-end training & evaluation.

    python -m cropyield.train

Steps
 1. Load dataset (builds the synthetic one if missing) + feature engineering
 2. Early fusion: compare Ridge / Random Forest / XGBoost / LightGBM / MLP with LOYO
 3. Late fusion: per-modality boosters + stacked meta-learner, same LOYO protocol
 4. Modality ablation: every combination of Satellite / Weather / Soil
 5. Per-year, per-crop, per-state error + spatial (leave-districts-out) check
 6. Fit the deployed model on all data; SHAP contributions -> feature & modality importance
 7. Prediction intervals from LOYO residuals; UI presets; figures; model bundle
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from itertools import combinations

import joblib
import numpy as np
import pandas as pd

from .config import (ALL_FEATURES, DATASET_PATH, FEATURE_INFO, FIGURE_DIR, MODALITIES, MODEL_BUNDLE_PATH,
                     MODEL_DIR, REPORT_DIR, SATELLITE_RAW, SOIL_RAW, TARGET, WEATHER_RAW, modality_of)
from .explain import contributions
from .features import build_features, modality_columns
from .fusion import LateFusionRegressor
from .models import EXPLAINABLE, booster, make_models
from .validation import grouped_metrics, leave_districts_out, loyo, metrics

RAW_INPUTS = SATELLITE_RAW + WEATHER_RAW + SOIL_RAW


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        from .build_dataset import build
        log("Dataset not found -> building synthetic dataset")
        build("synthetic")
    return build_features(pd.read_csv(DATASET_PATH))


# --------------------------------------------------------------------------
def compare_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, predictions = [], {}
    for name, model in make_models().items():
        preds, seconds = loyo(df, ALL_FEATURES, model)
        predictions[name] = preds
        rows.append({"Model": name, "Fusion": "Early", **metrics(preds.actual, preds.predicted),
                     "CV_seconds": round(seconds, 1)})
        log(f"  {name:<22} RMSE={rows[-1]['RMSE']:.3f}  R2={rows[-1]['R2']:.3f}")

    late = LateFusionRegressor({m: modality_columns([m]) for m in MODALITIES}, booster())
    preds, seconds = loyo(df, ALL_FEATURES, late)
    name = "Late Fusion (stacked)"
    predictions[name] = preds
    rows.append({"Model": name, "Fusion": "Late", **metrics(preds.actual, preds.predicted),
                 "CV_seconds": round(seconds, 1)})
    log(f"  {name:<22} RMSE={rows[-1]['RMSE']:.3f}  R2={rows[-1]['R2']:.3f}")
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True), predictions


def modality_ablation(df: pd.DataFrame, model) -> pd.DataFrame:
    rows = []
    names = list(MODALITIES)
    for k in (1, 2, 3):
        for combo in combinations(names, k):
            preds, _ = loyo(df, modality_columns(list(combo)), model)
            label = " + ".join(combo) if k < 3 else "All three (early fusion)"
            rows.append({"Modalities": label, "n_modalities": k, **metrics(preds.actual, preds.predicted)})
    return pd.DataFrame(rows)


def prediction_intervals(preds: pd.DataFrame) -> dict[str, dict[str, float]]:
    """80% empirical interval from out-of-sample (LOYO) residuals, per crop."""
    out = {}
    for crop, g in preds.groupby("crop"):
        residual = g["actual"] - g["predicted"]
        out[crop] = {"lower": float(residual.quantile(0.10)), "upper": float(residual.quantile(0.90))}
    return out


def ui_presets(df: pd.DataFrame) -> dict:
    """Typical input values per district/crop so the predict form can be pre-filled."""
    presets = {}
    for (district, crop), g in df.groupby(["district", "crop"]):
        presets[f"{district}|{crop}"] = {
            "district": district, "crop": crop, "state": g["state"].iloc[0],
            "lat": float(g["lat"].iloc[0]), "lon": float(g["lon"].iloc[0]),
            "mean_yield": float(g[TARGET].mean()),
            "inputs": {c: round(float(g[c].median()), 4) for c in RAW_INPUTS},
        }
    crop_defaults = {crop: {c: round(float(g[c].median()), 4) for c in RAW_INPUTS}
                     for crop, g in df.groupby("crop")}
    ranges = {c: {"p01": float(df[c].quantile(0.01)), "p99": float(df[c].quantile(0.99)),
                  "min": float(df[c].min()), "max": float(df[c].max())} for c in RAW_INPUTS}
    return {"presets": presets, "crop_defaults": crop_defaults, "ranges": ranges}


# --------------------------------------------------------------------------
def save_figures(comparison, ablation, best_preds, importance, modality_share, deployed) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
    green, grey = "#2f7d4f", "#b8c4bb"

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    colors = [green if m == deployed else grey for m in comparison["Model"]]
    ax.barh(comparison["Model"][::-1], comparison["RMSE"][::-1], color=colors[::-1])
    ax.set_xlabel("LOYO RMSE (t/ha) - lower is better")
    ax.set_title("Model comparison (leave-one-year-out)")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "model_comparison.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ab = ablation.sort_values("R2")
    ax.barh(ab["Modalities"], ab["R2"], color=[green if n == 3 else grey for n in ab["n_modalities"]])
    ax.set_xlabel("LOYO R² - higher is better")
    ax.set_title("Modality ablation: fusion beats every single source")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "modality_ablation.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for crop, color in (("Wheat", "#c48a1c"), ("Rice", green)):
        g = best_preds[best_preds.crop == crop]
        ax.scatter(g["actual"], g["predicted"], s=12, alpha=0.65, label=crop, color=color)
    lim = [best_preds[["actual", "predicted"]].min().min() - 0.2, best_preds[["actual", "predicted"]].max().max() + 0.2]
    ax.plot(lim, lim, "--", color="#555", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual yield (t/ha)"); ax.set_ylabel("Predicted yield (t/ha)")
    ax.set_title(f"{deployed}: LOYO predictions"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "predicted_vs_actual.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    top = importance.head(15)[::-1]
    palette = {"Satellite": green, "Weather": "#3b78b5", "Soil": "#9a6a3a", "Cross-modal": "#8a5bb0",
               "Context": "#777"}
    ax.barh(top["label"], top["mean_abs_shap"], color=[palette[m] for m in top["modality"]])
    ax.set_xlabel("Mean |SHAP value| (t/ha)")
    ax.set_title("Top features by SHAP importance")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "shap_importance.png"); plt.close(fig)

    modality_share = modality_share[modality_share["share_pct"] >= 0.5]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pie(modality_share["share_pct"], labels=modality_share["modality"], autopct="%1.0f%%",
           colors=[palette[m] for m in modality_share["modality"]], startangle=90,
           wedgeprops={"width": 0.45, "edgecolor": "white"})
    ax.set_title("Share of SHAP importance by modality")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "modality_share.png"); plt.close(fig)


# --------------------------------------------------------------------------
def run() -> dict:
    started = time.perf_counter()
    df = load_dataset()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    source = str(df["weather_source"].iloc[0])
    log(f"Loaded {len(df)} rows | {df.district.nunique()} districts | years {df.year.min()}-{df.year.max()} "
        f"| weather: {source}")

    log("1/5 Early vs late fusion model comparison (LOYO)")
    comparison, all_preds = compare_models(df)
    explainable = comparison[comparison["Model"].isin(EXPLAINABLE)]
    deployed = explainable.iloc[0]["Model"]
    best_overall = comparison.iloc[0]["Model"]
    deployed_preds = all_preds[deployed]
    log(f"   best overall: {best_overall} | deployed (best explainable): {deployed}")

    log("2/5 Modality ablation")
    deployed_template = make_models()[deployed]
    ablation = modality_ablation(df, deployed_template)

    log("3/5 Per-year / per-crop / spatial validation")
    per_year = grouped_metrics(deployed_preds, "year")
    per_crop = grouped_metrics(deployed_preds, "crop")
    per_state = grouped_metrics(deployed_preds, "state").sort_values("RMSE")
    spatial_rows = []
    for name in ("Ridge Regression", deployed):
        p = leave_districts_out(df, ALL_FEATURES, make_models()[name])
        spatial_rows.append({"Model": name, "Protocol": "Leave-districts-out (8 folds)",
                             **metrics(p.actual, p.predicted)})
    spatial = pd.DataFrame(spatial_rows).drop_duplicates("Model")

    log("4/5 Fit deployed model on all data + SHAP explainability")
    final_model = make_models()[deployed].fit(df[ALL_FEATURES], df[TARGET])
    contrib, base = contributions(final_model, df[ALL_FEATURES])
    importance = (contrib.abs().mean().rename("mean_abs_shap").reset_index()
                  .rename(columns={"index": "feature"}).sort_values("mean_abs_shap", ascending=False))
    importance["modality"] = importance["feature"].map(modality_of)
    importance["label"] = importance["feature"].map(lambda f: FEATURE_INFO[f]["label"])
    modality_share = importance.groupby("modality", as_index=False)["mean_abs_shap"].sum()
    modality_share["share_pct"] = 100 * modality_share["mean_abs_shap"] / modality_share["mean_abs_shap"].sum()
    modality_share = modality_share.sort_values("share_pct", ascending=False)

    late = LateFusionRegressor({m: modality_columns([m]) for m in MODALITIES}, booster())
    late.fit(df[ALL_FEATURES], df[TARGET], groups=df["year"].to_numpy())
    late_weights = late.modality_weights

    log("5/5 Saving reports, figures and model bundle")
    intervals = prediction_intervals(deployed_preds)
    deployed_row = comparison[comparison["Model"] == deployed].iloc[0]
    summary = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": {"weather": source, "satellite": "simulated NDVI composites",
                        "soil": "simulated Soil-Health-Card style samples", "yield": "simulated"},
        "rows": int(len(df)), "districts": int(df.district.nunique()), "states": int(df.state.nunique()),
        "crops": sorted(df.crop.unique().tolist()), "years": sorted(map(int, df.year.unique())),
        "features": ALL_FEATURES, "n_features": len(ALL_FEATURES),
        "best_model_overall": best_overall, "deployed_model": deployed,
        "deployed_metrics": {k: float(deployed_row[k]) for k in ("RMSE", "MAE", "R2", "MAPE_pct")},
        "fused_vs_best_single_r2_gain": float(
            ablation.loc[ablation.n_modalities == 3, "R2"].iloc[0] - ablation.loc[ablation.n_modalities == 1, "R2"].max()),
        "late_fusion_weights": late_weights,
        "prediction_interval_80": intervals,
        "training_seconds": round(time.perf_counter() - started, 1),
    }

    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    ablation.to_csv(REPORT_DIR / "modality_ablation.csv", index=False)
    per_year.to_csv(REPORT_DIR / "per_year_metrics.csv", index=False)
    per_crop.to_csv(REPORT_DIR / "per_crop_metrics.csv", index=False)
    per_state.to_csv(REPORT_DIR / "per_state_metrics.csv", index=False)
    spatial.to_csv(REPORT_DIR / "spatial_validation.csv", index=False)
    importance.to_csv(REPORT_DIR / "feature_importance_shap.csv", index=False)
    modality_share.to_csv(REPORT_DIR / "modality_contribution.csv", index=False)
    pd.concat([p.assign(model=m) for m, p in all_preds.items()]).to_csv(REPORT_DIR / "loyo_predictions.csv", index=False)
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    save_figures(comparison, ablation, deployed_preds, importance, modality_share, deployed)

    bundle = {
        "model": final_model, "model_name": deployed, "features": ALL_FEATURES, "raw_inputs": RAW_INPUTS,
        "intervals": intervals, **ui_presets(df), "summary": summary,
        "mean_base_value": float(np.mean(base)),
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    print("\n" + comparison.round(3).to_string(index=False))
    print("\nModality ablation (deployed model):\n" + ablation.round(3).to_string(index=False))
    print(f"\nDeployed model: {deployed} -> {MODEL_BUNDLE_PATH}")
    log(f"Finished in {summary['training_seconds']} s")
    return summary


if __name__ == "__main__":
    run()
