"""Candidate regression models (PPT slide 8).

All models are sklearn Pipelines so imputation/scaling is learned inside each
LOYO training fold only (no leakage from the held-out year).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED

warnings.filterwarnings("ignore", category=UserWarning)

# Models whose predictions we can explain exactly with SHAP-style contributions
EXPLAINABLE = {"Ridge Regression", "Random Forest", "XGBoost", "LightGBM"}


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median")


def ridge() -> Pipeline:
    return Pipeline([("imputer", _imputer()), ("scaler", StandardScaler()),
                     ("model", RidgeCV(alphas=np.logspace(-2, 3, 30)))])


def random_forest() -> Pipeline:
    return Pipeline([("imputer", _imputer()),
                     ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5,
                                                     n_jobs=-1, random_state=RANDOM_SEED))])


def xgboost_model() -> Pipeline | None:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return Pipeline([("imputer", _imputer()),
                     ("model", XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
                                            colsample_bytree=0.7, min_child_weight=3, reg_lambda=2.0,
                                            objective="reg:squarederror", n_jobs=4, random_state=RANDOM_SEED))])


def lightgbm_model() -> Pipeline | None:
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return None
    return Pipeline([("imputer", _imputer()),
                     ("model", LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                                             colsample_bytree=0.7, reg_lambda=1.0, verbose=-1,
                                             random_state=RANDOM_SEED))])


def mlp() -> Pipeline:
    return Pipeline([("imputer", _imputer()), ("scaler", StandardScaler()),
                     ("model", MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
                                            max_iter=3000, early_stopping=True, validation_fraction=0.15,
                                            n_iter_no_change=40, random_state=RANDOM_SEED))])


def make_models() -> dict[str, Pipeline]:
    """Ordered dict of available early-fusion models."""
    candidates = {
        "Ridge Regression": ridge(),
        "Random Forest": random_forest(),
        "XGBoost": xgboost_model(),
        "LightGBM": lightgbm_model(),
        "Neural Net (MLP)": mlp(),
    }
    return {name: model for name, model in candidates.items() if model is not None}


def booster() -> Pipeline:
    """Default tree booster used for ablation / late-fusion base learners."""
    return lightgbm_model() or xgboost_model() or random_forest()
