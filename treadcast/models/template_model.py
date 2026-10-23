"""Learned features: a 1D CNN that reads the shape of the contact-patch template.

Input: the daily template (256 points, normalised so 1 = off the road) plus the operating
conditions (speed, pressure, temperature, load, axle, new depth). Mileage is left out on
purpose, so the CNN has to learn depth from the signal shape. Its output then goes into
the gradient-boosted model as one more feature (stacking, with out-of-fold predictions).

PyTorch is used when installed. Without it (for example in the light app image) the same
interface falls back to PCA of the template + gradient boosting, so the pipeline still runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CNN_CONTEXT = ["speed_kmh_meas", "pressure_bar", "tire_temp_c", "load_est_kg", "is_steer", "new_depth_mm"]

try:                                                    # optional dependency
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:                                     # pragma: no cover
    HAS_TORCH = False


def backend() -> str:
    return "cnn" if HAS_TORCH else "pca-gbm"


if HAS_TORCH:
    class _Net(nn.Module):
        def __init__(self, n_ctx: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, 9, padding=4), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 48, 5, padding=2), nn.BatchNorm1d(48), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(48, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
            )
            self.head = nn.Sequential(nn.Linear(64 * 4 + n_ctx, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

        def forward(self, x, ctx):
            h = self.conv(x.unsqueeze(1)).flatten(1)
            return self.head(torch.cat([h, ctx], dim=1)).squeeze(1)


class TemplateModel:
    """fit(templates, table) on labelled rows; predict(templates, table) -> depth in mm."""

    def __init__(self, epochs: int = 25, batch: int = 256, lr: float = 2e-3, seed: int = 0, use_torch: bool | None = None):
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed
        self.use_torch = HAS_TORCH if use_torch is None else (use_torch and HAS_TORCH)

    # --- scaling helpers ---
    def _scale(self, tpl, df):
        x = (tpl - self.t_mean_) / self.t_std_
        c = (df[CNN_CONTEXT].to_numpy(float) - self.c_mean_) / self.c_std_
        return x.astype(np.float32), c.astype(np.float32)

    def fit(self, tpl: np.ndarray, df: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> "TemplateModel":
        self.t_mean_, self.t_std_ = tpl.mean(axis=0), tpl.std(axis=0) + 1e-6
        ctx = df[CNN_CONTEXT].to_numpy(float)
        self.c_mean_, self.c_std_ = ctx.mean(axis=0), ctx.std(axis=0) + 1e-6
        self.y_mean_, self.y_std_ = float(np.mean(y)), float(np.std(y) + 1e-6)
        x, c = self._scale(tpl, df)
        yz = ((y - self.y_mean_) / self.y_std_).astype(np.float32)
        if self.use_torch:
            self._fit_torch(x, c, yz, w.astype(np.float32))
        else:
            self._fit_fallback(x, c, yz, w)
        return self

    def predict(self, tpl: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        x, c = self._scale(tpl, df)
        z = self._predict_torch(x, c) if self.use_torch else self.gbm_.predict(np.hstack([self.pca_.transform(x), c]))
        return z * self.y_std_ + self.y_mean_

    # --- PyTorch 1D CNN ---
    def _fit_torch(self, x, c, y, w):
        torch.manual_seed(self.seed)
        self.net_ = _Net(c.shape[1])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * int(np.ceil(len(x) / self.batch)))
        X, Cc, Y, W = map(torch.from_numpy, (x, c, y, w))
        gen = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(X), generator=gen)
            for i in range(0, len(X), self.batch):
                b = order[i:i + self.batch]
                if len(b) < 2:                          # BatchNorm needs at least 2 rows
                    continue
                pred = self.net_(X[b], Cc[b])
                loss = (W[b] * (pred - Y[b]) ** 2).sum() / W[b].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        self.net_.eval()

    def _predict_torch(self, x, c):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 2048):
                out.append(self.net_(torch.from_numpy(x[i:i + 2048]), torch.from_numpy(c[i:i + 2048])).numpy())
        return np.concatenate(out) if out else np.zeros(0)

    # --- fallback without PyTorch ---
    def _fit_fallback(self, x, c, y, w):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.pca_ = PCA(n_components=16, random_state=self.seed).fit(x)
        self.gbm_ = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=self.seed)
        self.gbm_.fit(np.hstack([self.pca_.transform(x), c]), y, sample_weight=w)

    def save(self, path: str) -> None:
        if self.use_torch:
            torch.save({"state": self.net_.state_dict(), "scales": self._scales()}, path)
        else:
            import joblib
            joblib.dump(self, path)

    def _scales(self):
        return dict(t_mean=self.t_mean_, t_std=self.t_std_, c_mean=self.c_mean_, c_std=self.c_std_,
                    y_mean=self.y_mean_, y_std=self.y_std_)
