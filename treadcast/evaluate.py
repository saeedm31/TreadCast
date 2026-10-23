"""Scores. Always on tires the models never saw (the test trucks)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def scores(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    err = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return dict(mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err ** 2))),
                r2=float(1 - np.sum(err ** 2) / ss) if ss > 0 else float("nan"),
                bias=float(np.mean(err)), n=int(m.sum()))


def by_group(df: pd.DataFrame, pred: str, group: str, truth: str = "depth_true") -> list[dict]:
    out = []
    for key, d in df.groupby(group, observed=True):
        s = scores(d[truth], d[pred])
        out.append({group: str(key), "mae": round(s["mae"], 3), "n": s["n"]})
    return out


def remaining_scores(df: pd.DataFrame) -> dict:
    """Km-left error where it matters: tires between 2 mm above the limit and 3 mm below new."""
    m = (df["depth_true"] > C.REPLACE_AT_MM + 2) & (df["depth_true"] < df["new_depth_mm"] - 3)
    d = df[m]
    t = d["remaining_true_km"].to_numpy()
    est, safe = d["remaining_km"].to_numpy(), d["remaining_km_safe"].to_numpy()
    rel = np.abs(est - t) / np.maximum(t, 1.0)
    return dict(
        n=int(len(d)),
        mae_km=float(np.mean(np.abs(est - t))),
        median_rel_error=float(np.median(rel)),
        over_estimate_rate=float(np.mean(est > t)),
        safe_over_estimate_rate=float(np.mean(safe > t)),
        safe_mean_margin_km=float(np.mean(t - safe)),
    )


def safety_scores(df: pd.DataFrame) -> dict:
    """Days a tire drove below the fleet limit, and whether each policy had flagged it."""
    below = df[df["depth_true"] < C.REPLACE_AT_MM]
    if below.empty:
        return dict(days_below_limit=0)
    flagged_tracker = below["remaining_km_safe"] <= 0
    flagged_service = below["last_gauge_mm"] <= C.REPLACE_AT_MM + 1.0
    return dict(days_below_limit=int(len(below)),
                tracker_flagged_share=float(flagged_tracker.mean()),
                service_only_flagged_share=float(flagged_service.mean()))
