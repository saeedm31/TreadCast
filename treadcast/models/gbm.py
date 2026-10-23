"""Gradient-boosted trees on the fused features (physics + signal + context + CNN output).

scikit-learn's HistGradientBoostingRegressor: the same family as XGBoost / LightGBM, with
native missing values (GPS gaps) and monotone constraints, and no extra dependency.
Mileage and age may only push the depth down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MONOTONE_DOWN = {"km_since_install", "days_since_install"}
CATEGORICAL = {"sku_code"}


class DepthGBM:
    def __init__(self, features: list[str], seed: int = 0):
        self.features = list(features)
        self.model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, random_state=seed,
            monotonic_cst=[-1 if f in MONOTONE_DOWN else 0 for f in self.features],
            categorical_features=[f in CATEGORICAL for f in self.features],
        )

    def fit(self, df: pd.DataFrame) -> "DepthGBM":
        lab = df[df["label_weight"] > 0]
        self.model.fit(lab[self.features], lab["label"], sample_weight=lab["label_weight"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features])
