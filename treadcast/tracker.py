"""Per-tire tracker: turns noisy daily estimates into one depth, a wear rate and a range.

A Kalman filter with two numbers per tire: depth (mm) and wear rate (mm per 1000 km).
- Each driving day, depth goes down by wear rate x km driven (the prediction step).
- The day's model estimate pulls the depth towards it, weighted by how noisy the model is.
- A new tire starts at its known new depth; the wear rate starts at the fleet average.
The rate can't go below zero, so the depth can only go down over time.

Remaining km = (depth - limit) / wear rate. The safe value uses the low end of the depth
range and the high end of the rate range, so it errs towards replacing too early.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

Z = 1.64                       # one-sided 95% range


def track(df: pd.DataFrame, estimate: str, model_sd: float, prior_rate: dict,
          limit: float = C.REPLACE_AT_MM) -> pd.DataFrame:
    """df: one row per tire per driving day with `estimate`, km_day, axle, new_depth_mm.
    Returns depth_kf, depth_sd, rate_kf, rate_sd, remaining_km, remaining_km_safe (same index)."""
    out = pd.DataFrame(index=df.index, columns=["depth_kf", "depth_sd", "rate_kf", "rate_sd",
                                                "remaining_km", "remaining_km_safe"], dtype=float)
    R = model_sd ** 2
    for _, rows in df.sort_values(["tire_id", "day"]).groupby("tire_id", sort=False):
        first = rows.iloc[0]
        r0 = prior_rate[first["axle"]]
        x = np.array([first["new_depth_mm"], r0])
        P = np.diag([0.2 ** 2, (0.35 * r0) ** 2])
        res = np.empty((len(rows), 6))
        for i, (km, z) in enumerate(zip(rows["km_day"].to_numpy(), rows[estimate].to_numpy())):
            k = km / 1000.0
            F = np.array([[1.0, -k], [0.0, 1.0]])
            Q = np.diag([(0.02 * k) ** 2, (0.0015 ** 2) * k])
            x = F @ x
            P = F @ P @ F.T + Q
            if np.isfinite(z):
                H = np.array([1.0, 0.0])
                S = H @ P @ H + R
                K = P @ H / S
                x = x + K * (z - x[0])
                P = P - np.outer(K, H @ P)
            x[1] = max(x[1], 0.005)
            d_sd, r_sd = np.sqrt(max(P[0, 0], 1e-6)), np.sqrt(max(P[1, 1], 1e-8))
            rem = max(x[0] - limit, 0.0) / x[1] * 1000.0
            rem_safe = max(x[0] - Z * d_sd - limit, 0.0) / (x[1] + Z * r_sd) * 1000.0
            res[i] = [x[0], d_sd, x[1], r_sd, rem, rem_safe]
        res[:, 0] = np.minimum.accumulate(res[:, 0])       # shown depth never goes back up
        out.loc[rows.index] = res
    return out.astype(float)


def true_remaining_km(df: pd.DataFrame, tires: pd.DataFrame, limit: float = C.REPLACE_AT_MM) -> np.ndarray:
    """Reference for scoring: km left until the limit at the tire's own average lifetime wear rate
    (the simulator knows it; a real fleet would only know it after the tire is removed)."""
    t = tires.set_index("tire_id")
    life_rate = (t["new_depth_mm"] - t["final_depth"]) / (t["km"].clip(lower=1.0) / 1000.0)
    rate = df["tire_id"].map(life_rate).to_numpy()
    return np.maximum(df["depth_true"].to_numpy() - limit, 0.0) / np.maximum(rate, 1e-3) * 1000.0
