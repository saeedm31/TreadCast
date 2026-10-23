"""The model to beat: depth from mileage only.

depth = new depth - average wear rate x km. One wear rate per axle (steer, drive),
fitted on the training labels. No sensor data at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MileageBaseline:
    def fit(self, df: pd.DataFrame) -> "MileageBaseline":
        lab = df[df["label_weight"] > 0]
        self.rate_ = {}
        for axle, d in lab.groupby("axle"):
            km = d["km_since_install"].to_numpy() / 1000.0
            worn = d["new_depth_mm"].to_numpy() - d["label"].to_numpy()
            w = d["label_weight"].to_numpy()
            self.rate_[axle] = float(np.sum(w * km * worn) / np.sum(w * km * km))    # least squares through 0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        rate = df["axle"].map(self.rate_).to_numpy()
        return df["new_depth_mm"].to_numpy() - rate * df["km_since_install"].to_numpy() / 1000.0
