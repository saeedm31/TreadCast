"""Training labels from the few real measurements a fleet has.

Real depth is known only at fitting (a new tire's depth is on its spec sheet) and when a
technician gauges the tire at a service, a few times per tire life. Between two known
points we fill in labels by straight-line interpolation over km (pseudo-labels) and give
them a lower weight. Rows after the last known point get no label: we never extrapolate.

Pseudo-labels use a later measurement, so they are only for training. Scores are always
computed against real readings (or, in this simulation, the true depth).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def attach_labels(table: pd.DataFrame, gauges: pd.DataFrame, max_day: int | None = None) -> pd.DataFrame:
    """Adds label, label_weight and is_gauge_day. Only gauges taken before max_day are used."""
    g = gauges if max_day is None else gauges[gauges["day"] < max_day]
    out = table.copy()
    out["label"] = np.nan
    out["label_weight"] = 0.0
    out["is_gauge_day"] = False
    for tire_id, rows in out.groupby("tire_id"):
        pts = g[g["tire_id"] == tire_id].sort_values("km_since_install")
        if len(pts) < 2:
            continue
        km = rows["km_since_install"].to_numpy()
        inside = (km >= pts["km_since_install"].min()) & (km <= pts["km_since_install"].max())
        lab = np.interp(km, pts["km_since_install"], pts["depth_gauge"])
        idx = rows.index[inside]
        out.loc[idx, "label"] = lab[inside]
        out.loc[idx, "label_weight"] = C.PSEUDO_LABEL_WEIGHT
        # the last driving day before each service carries the real reading
        for _, p in pts[~pts["is_new"]].iterrows():
            before = rows[rows["day"] <= p["day"]]
            if len(before):
                i = before.index[-1]
                out.loc[i, ["label", "label_weight", "is_gauge_day"]] = [p["depth_gauge"], 1.0, True]
    return out
