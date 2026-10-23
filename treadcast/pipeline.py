"""End to end: simulate -> features -> labels -> models -> tracker -> scores -> artifacts.

    python -m treadcast                    # full run, writes artifacts/
    python -m treadcast --fast             # small, quick run (tests, CI smoke)
"""

from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config as C
from .evaluate import by_group, remaining_scores, safety_scores, scores
from .features import CONTEXT, PHYSICS, SIGNAL, build_dataset
from .labels import attach_labels
from .models.baseline import MileageBaseline
from .models.gbm import DepthGBM
from .models.template_model import TemplateModel, backend
from .sim.fleet import simulate_fleet
from .tracker import track, true_remaining_km

ABLATION = {
    "context only": CONTEXT,
    "+ physics": CONTEXT + PHYSICS,
    "+ signal": CONTEXT + PHYSICS + SIGNAL,
    "+ CNN (final)": CONTEXT + PHYSICS + SIGNAL + ["cnn_depth"],
}
FINAL = "+ CNN (final)"


def _log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def run(out_dir: str = "artifacts", days: int = C.DAYS, seed: int = C.SEED, epochs: int = 25,
        folds: int = 4) -> dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    rows, gauges, tires = simulate_fleet(seed=seed, days=days)
    _log(f"simulated {len(tires)} tires, {len(rows)} tire-days, {len(gauges)} gauge readings", t0)
    table, templates = build_dataset(rows, tires, seed=seed)
    _log("features done", t0)

    cutoff = min(C.TRAIN_CUTOFF_DAY, int(days * 0.7))
    table["split"] = np.where(table["truck"].isin(C.TEST_TRUCKS), "test", "train")
    train_mask = (table["split"] == "train") & (table["day"] < cutoff)
    lab_train = attach_labels(table[train_mask], gauges, max_day=cutoff)
    lab_train = lab_train[lab_train["label_weight"] > 0]
    _log(f"{len(lab_train)} labelled training rows ({int(lab_train['label_weight'].eq(1).sum())} real readings)", t0)

    # baseline
    base = MileageBaseline().fit(lab_train)
    table["pred_baseline"] = base.predict(table)

    # CNN on the template: out-of-fold for the training rows, full model for the rest
    table["cnn_depth"] = np.nan
    gkf = GroupKFold(n_splits=folds)
    tr_idx = lab_train.index.to_numpy()
    for k, (a, b) in enumerate(gkf.split(tr_idx, groups=lab_train["tire_id"])):
        ia, ib = tr_idx[a], tr_idx[b]
        m = TemplateModel(epochs=epochs, seed=seed + k).fit(
            templates[ia], table.loc[ia], lab_train.loc[ia, "label"].to_numpy(), lab_train.loc[ia, "label_weight"].to_numpy())
        table.loc[ib, "cnn_depth"] = m.predict(templates[ib], table.loc[ib])
        _log(f"template model ({backend()}) fold {k + 1}/{folds}", t0)
    cnn = TemplateModel(epochs=epochs, seed=seed).fit(
        templates[tr_idx], table.loc[tr_idx], lab_train["label"].to_numpy(), lab_train["label_weight"].to_numpy())
    rest = table.index.difference(tr_idx)
    table.loc[rest, "cnn_depth"] = cnn.predict(templates[rest], table.loc[rest])
    _log("template model full fit", t0)

    # gradient boosting: ablation from context only up to the final fused model
    lab_train["cnn_depth"] = table.loc[lab_train.index, "cnn_depth"]
    models = {}
    for name, feats in ABLATION.items():
        models[name] = DepthGBM(feats, seed=seed).fit(lab_train)
        table[f"pred_{_slug(name)}"] = models[name].predict(table)
    _log("gradient boosting models done", t0)

    # noise of the final model at real readings, from out-of-fold predictions
    oof = pd.Series(np.nan, index=lab_train.index)
    for a, b in gkf.split(tr_idx, groups=lab_train["tire_id"]):
        g = DepthGBM(ABLATION[FINAL], seed=seed).fit(lab_train.iloc[a])
        oof.iloc[b] = g.predict(lab_train.iloc[b])
    real = lab_train["label_weight"] == 1.0
    model_sd = float(np.sqrt(max(np.mean((oof[real] - lab_train.loc[real, "label"]) ** 2) - C.GAUGE_NOISE_MM ** 2, 0.05)))
    _log(f"model noise at real readings: {model_sd:.2f} mm", t0)

    # tracker for every tire
    final_col = f"pred_{_slug(FINAL)}"
    tr = track(table, final_col, model_sd, base.rate_)
    table = table.join(tr)
    table["remaining_true_km"] = true_remaining_km(table, tires)
    table["km_per_day_30"] = (table.sort_values("day").groupby("tire_id")["km_day"]
                              .transform(lambda s: s.rolling(30, min_periods=1).mean()))
    table = _attach_last_gauge(table, gauges)
    _log("tracker done", t0)

    metrics = _metrics(table, gauges, final_col, model_sd, base, cutoff, len(lab_train), int(real.sum()))
    _save(out_dir, table, gauges, tires, templates, metrics, models[FINAL], cnn)
    _log(f"artifacts written to {out_dir}/", t0)
    return metrics


def _slug(name: str) -> str:
    return {"context only": "context", "+ physics": "physics", "+ signal": "signal", "+ CNN (final)": "final"}[name]


def _attach_last_gauge(table: pd.DataFrame, gauges: pd.DataFrame) -> pd.DataFrame:
    """The latest gauge reading each day: what a service-only fleet knows."""
    g = gauges.sort_values("day")[["tire_id", "day", "depth_gauge"]].rename(columns={"day": "gauge_day"})
    t = table.reset_index().sort_values("day")
    merged = pd.merge_asof(t, g, left_on="day", right_on="gauge_day", by="tire_id", direction="backward")
    merged = merged.set_index("index").sort_index()
    table["last_gauge_mm"] = merged["depth_gauge"]
    return table


def _metrics(table, gauges, final_col, model_sd, base, cutoff, n_lab, n_real) -> dict:
    test = table[table["split"] == "test"].copy()
    test["season"] = test["date"].dt.quarter.map({1: "winter", 2: "spring", 3: "summer", 4: "autumn"})
    test["speed_band"] = pd.cut(test["speed_kmh"], [0, 50, 70, 200], labels=["< 50 km/h", "50–70 km/h", "> 70 km/h"])
    models = {"Mileage baseline": "pred_baseline", "1D CNN alone": "cnn_depth"}
    models.update({f"GBM {k}": f"pred_{_slug(k)}" for k in ABLATION})
    models["Tracker (final)"] = "depth_kf"
    at_gauge = attach_labels(test, gauges)
    at_gauge = at_gauge[at_gauge["is_gauge_day"]]
    return dict(
        backend=backend(),
        model_noise_mm=model_sd,
        baseline_rates_mm_per_1000km=base.rate_,
        train=dict(cutoff_day=cutoff, labelled_rows=n_lab, real_readings=n_real),
        test=dict(trucks=C.TEST_TRUCKS, rows=int(len(test)), tires=int(test["tire_id"].nunique())),
        depth=[{"model": k, **scores(test["depth_true"], test[v])} for k, v in models.items()],
        depth_after_cutoff=[{"model": k, **scores(test.loc[test["day"] >= cutoff, "depth_true"], test.loc[test["day"] >= cutoff, v])}
                            for k, v in models.items()],
        by_axle=by_group(test, "depth_kf", "axle"),
        by_route=by_group(test, "depth_kf", "route"),
        by_season=by_group(test, "depth_kf", "season"),
        by_speed=by_group(test, final_col, "speed_band"),
        remaining=remaining_scores(test),
        safety=safety_scores(test),
        gauge_check={k: scores(at_gauge["label"], at_gauge[v]) for k, v in
                     [("Mileage baseline", "pred_baseline"), ("Tracker (final)", "depth_kf")]},
    )


def _save(out_dir, table, gauges, tires, templates, metrics, gbm, cnn):
    cols = ["day", "date", "truck", "route", "position", "axle", "tire_id", "sku", "split", "km_day",
            "km_since_install", "speed_kmh", "pressure_bar", "tire_temp_c", "load_frac", "depth_true",
            "pred_baseline", "cnn_depth", "pred_context", "pred_physics", "pred_signal", "pred_final",
            "depth_kf", "depth_sd", "rate_kf", "rate_sd", "remaining_km", "remaining_km_safe",
            "remaining_true_km", "km_per_day_30", "last_gauge_mm"]
    out = table[cols].copy()
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(3)
    out.to_csv(os.path.join(out_dir, "predictions.csv.gz"), index=False)
    gauges.to_csv(os.path.join(out_dir, "gauges.csv"), index=False)
    tires.to_csv(os.path.join(out_dir, "tires.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=float)
    # a few real templates per test tire for the app: new, half worn, worn
    pick = []
    for _, d in table[table["split"] == "test"].groupby("tire_id"):
        d = d.sort_values("day")
        pick += [d.index[0], d.index[len(d) // 2], d.index[-1]]
    np.savez_compressed(os.path.join(out_dir, "examples.npz"), index=np.array(pick),
                        templates=templates[pick].astype(np.float16),
                        tire_id=table.loc[pick, "tire_id"].to_numpy().astype(str),
                        depth=table.loc[pick, "depth_true"].to_numpy(), day=table.loc[pick, "day"].to_numpy())
    joblib.dump(gbm, os.path.join(out_dir, "gbm_final.joblib"), compress=3)
    cnn.save(os.path.join(out_dir, "template_model.pt" if cnn.use_torch else "template_model.joblib"))
