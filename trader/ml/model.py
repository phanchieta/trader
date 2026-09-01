"""LgbmSignal: gradient boosting (LightGBM) que devolve uma probabilidade de
"o retorno futuro supera o custo". Defaults bem regularizados — dado financeiro
overfitta fácil.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_DEFAULTS = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, max_depth=4,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                 min_child_samples=80, reg_lambda=1.0, random_state=0,
                 n_jobs=-1, verbosity=-1)


@dataclass
class LgbmSignal:
    params: dict = field(default_factory=lambda: dict(_DEFAULTS))
    model: object = None
    cols: list | None = None

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "LgbmSignal":
        from lightgbm import LGBMClassifier

        self.cols = list(X.columns)
        self.model = LGBMClassifier(**self.params)
        self.model.fit(X.to_numpy(dtype="float64"), np.asarray(y).astype("int8"),
                       sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(
            X[self.cols].to_numpy(dtype="float64"))[:, 1]

    def importances(self) -> dict:
        return dict(sorted(zip(self.cols, self.model.feature_importances_),
                           key=lambda kv: -kv[1]))
