"""Validation protocols and metrics (PPT slides 9-10).

* Leave-One-Year-Out (LOYO): hold out an entire season, train on the rest.
  Mirrors real forecasting of an unseen season and avoids temporal leakage.
* Leave-Districts-Out (spatial CV): hold out groups of whole districts to test
  generalisation to places the model has never seen.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .config import TARGET

ID_COLS = ["district", "state", "crop", "year"]


def metrics(actual, predicted) -> dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "MAE": float(mean_absolute_error(a, p)),
        "R2": float(r2_score(a, p)) if len(a) > 1 else float("nan"),
        "MAPE_pct": float(np.mean(np.abs((a - p) / np.maximum(np.abs(a), 1e-6))) * 100),
    }


def fit_model(model, X: pd.DataFrame, y: pd.Series, groups=None):
    if getattr(model, "uses_groups", False):
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


def _grouped_cv(df: pd.DataFrame, cols: list[str], model, splits) -> tuple[pd.DataFrame, float]:
    rows = []
    started = time.perf_counter()
    for train_idx, test_idx in splits:
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        fitted = fit_model(clone(model), train[cols], train[TARGET], groups=train["year"].to_numpy())
        out = test[ID_COLS].copy()
        out["actual"] = test[TARGET].to_numpy()
        out["predicted"] = fitted.predict(test[cols])
        rows.append(out)
    return pd.concat(rows).sort_index(), time.perf_counter() - started


def loyo(df: pd.DataFrame, cols: list[str], model) -> tuple[pd.DataFrame, float]:
    """Leave-one-year-out predictions for every row + total wall time."""
    years = df["year"].to_numpy()
    splits = [(np.where(years != y)[0], np.where(years == y)[0]) for y in np.unique(years)]
    return _grouped_cv(df, cols, model, splits)


def leave_districts_out(df: pd.DataFrame, cols: list[str], model, n_splits: int = 8) -> pd.DataFrame:
    splits = GroupKFold(n_splits=n_splits).split(df, groups=df["district"])
    return _grouped_cv(df, cols, model, splits)[0]


def grouped_metrics(predictions: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = [{by: key, "n": len(g), **metrics(g["actual"], g["predicted"])}
            for key, g in predictions.groupby(by)]
    return pd.DataFrame(rows)
