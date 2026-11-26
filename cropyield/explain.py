"""Explainability (PPT slide 10): exact per-feature contributions (SHAP values).

For tree boosters we use the libraries' built-in TreeSHAP
(`pred_contribs` / `pred_contrib`), so no extra dependency is needed.
Random Forest uses the `shap` package if installed. For Ridge the linear SHAP
value is simply coefficient x standardised feature value.

prediction = base_value + sum(contributions)   (exact additive decomposition)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def contributions(pipeline: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (per-feature contributions, base values) for each row of X."""
    estimator = pipeline[-1]
    Xt = pipeline[:-1].transform(X)
    Xt = np.asarray(Xt, dtype=float)
    kind = type(estimator).__name__

    if kind == "XGBRegressor":
        import xgboost as xgb
        raw = estimator.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
        values, base = raw[:, :-1], raw[:, -1]
    elif kind == "LGBMRegressor":
        raw = estimator.predict(Xt, pred_contrib=True)
        values, base = raw[:, :-1], raw[:, -1]
    elif kind in {"RidgeCV", "Ridge", "LinearRegression"}:
        values = Xt * estimator.coef_  # scaler output has zero mean on training data
        base = np.full(len(Xt), float(estimator.intercept_))
    elif kind == "RandomForestRegressor":
        import shap  # optional dependency
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(Xt, check_additivity=False)
        base = np.full(len(Xt), float(np.ravel(explainer.expected_value)[0]))
    else:
        raise TypeError(f"No exact explainer for {kind}")
    return pd.DataFrame(values, columns=list(X.columns), index=X.index), np.asarray(base, dtype=float)
