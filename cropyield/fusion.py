"""Late fusion (PPT slide 6): one model per modality, combined by a meta-learner.

    Satellite features --> model_S --\
    Weather features   --> model_W ----> meta-learner (non-negative linear) --> yield
    Soil features      --> model_So -/

The meta-learner is trained on OUT-OF-FOLD base predictions (inner
GroupKFold by year) so it never learns from predictions on rows the base
models were trained on - this is standard stacking without leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold


class LateFusionRegressor(BaseEstimator, RegressorMixin):
    uses_groups = True  # tells the validation loop to pass year groups to fit()

    def __init__(self, modality_columns: dict[str, list[str]], base_estimator, inner_splits: int = 5):
        self.modality_columns = modality_columns
        self.base_estimator = base_estimator
        self.inner_splits = inner_splits

    def _oof_predictions(self, X: pd.DataFrame, y: np.ndarray, groups) -> np.ndarray:
        oof = np.zeros((len(X), len(self.modality_columns)))
        if groups is not None and len(np.unique(groups)) >= self.inner_splits:
            splits = GroupKFold(n_splits=self.inner_splits).split(X, y, groups)
        else:
            splits = KFold(n_splits=self.inner_splits, shuffle=True, random_state=0).split(X)
        for train_idx, valid_idx in splits:
            for j, cols in enumerate(self.modality_columns.values()):
                model = clone(self.base_estimator).fit(X.iloc[train_idx][cols], y[train_idx])
                oof[valid_idx, j] = model.predict(X.iloc[valid_idx][cols])
        return oof

    def fit(self, X: pd.DataFrame, y, groups=None):
        y = np.asarray(y, dtype=float)
        meta_X = self._oof_predictions(X, y, groups)
        self.meta_ = LinearRegression(positive=True).fit(meta_X, y)
        self.base_models_ = {name: clone(self.base_estimator).fit(X[cols], y)
                             for name, cols in self.modality_columns.items()}
        return self

    def modality_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: model.predict(X[self.modality_columns[name]])
                             for name, model in self.base_models_.items()}, index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.meta_.predict(self.modality_predictions(X).to_numpy())

    @property
    def modality_weights(self) -> dict[str, float]:
        return dict(zip(self.modality_columns, map(float, self.meta_.coef_)))
