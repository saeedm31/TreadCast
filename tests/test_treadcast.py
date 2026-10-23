"""Fast checks of the simulator, features, labels, tracker and a small end-to-end run."""

import json

import numpy as np
import pytest

from treadcast import config as C
from treadcast.features import ALL_FEATURES, build_dataset
from treadcast.labels import attach_labels
from treadcast.sim.fleet import simulate_fleet
from treadcast.sim.signal import THETA, clean_template, tire_geometry
from treadcast.tracker import track


@pytest.fixture(scope="module")
def sim():
    return simulate_fleet(seed=3, days=160)


def test_depth_only_goes_down_and_stays_positive(sim):
    rows, gauges, tires = sim
    for _, d in rows.groupby("tire_id"):
        assert np.all(np.diff(d.sort_values("day")["depth_true"].to_numpy()) <= 1e-9)
    assert rows["depth_true"].min() > 0
    assert set(rows["axle"]) == {"steer", "drive"}
    assert gauges["is_new"].sum() == len(tires)            # every tire starts with a known depth


def test_wear_changes_the_signal_the_expected_way():
    new, worn = clean_template([20.0, 4.0], 21.0, 80.0, 2800.0, 8.8, 45.0, 540.0, 1.0)[0]
    _, _, r_new, phi_new = tire_geometry(20.0, 21.0, 8.8, 2800.0, 1.0)
    _, _, r_worn, phi_worn = tire_geometry(4.0, 21.0, 8.8, 2800.0, 1.0)
    assert r_worn < r_new                                    # smaller rolling radius
    assert phi_worn < phi_new                                # shorter contact patch
    assert worn.max() / worn[0] > new.max() / new[0]         # taller spikes, relative to the level
    assert np.min(new[np.abs(THETA) < 5]) < 0.1 * new[0]     # ~0 g inside the contact patch


def test_features_are_finite_where_expected(sim):
    rows, gauges, tires = sim
    table, tpl = build_dataset(rows.head(400), tires, seed=3)
    assert tpl.shape == (400, C.TEMPLATE_POINTS)
    feats = table[ALL_FEATURES].drop(columns=["r_roll_mm"])  # r_roll_mm is NaN on GPS-less days
    assert np.isfinite(feats.to_numpy(float)).all()
    assert table["r_roll_mm"].isna().mean() < 0.4


def test_labels_never_extrapolate_and_real_readings_weigh_more(sim):
    rows, gauges, tires = sim
    lab = attach_labels(rows, gauges)
    got = lab[lab["label_weight"] > 0]
    assert set(got["label_weight"].round(2)) <= {C.PSEUDO_LABEL_WEIGHT, 1.0}
    for tire_id, d in got.groupby("tire_id"):
        last_km = gauges[gauges["tire_id"] == tire_id]["km_since_install"].max()
        assert d["km_since_install"].max() <= last_km + 1e-6


def test_tracker_smooths_noise_and_never_goes_up(sim):
    rows, gauges, tires = sim
    d = rows[rows["tire_id"] == rows["tire_id"].iloc[0]].copy()
    rng = np.random.default_rng(0)
    d["est"] = d["depth_true"] + rng.normal(0, 1.0, len(d))
    out = track(d, "est", 1.0, {"steer": 0.07, "drive": 0.085})
    assert np.all(np.diff(out["depth_kf"].to_numpy()) <= 1e-9)
    assert np.mean(np.abs(out["depth_kf"] - d["depth_true"])) < np.mean(np.abs(d["est"] - d["depth_true"]))
    assert (out["remaining_km_safe"] <= out["remaining_km"] + 1e-6).all()


def test_small_end_to_end_run(tmp_path):
    from treadcast.pipeline import run
    m = run(str(tmp_path), days=200, epochs=2, folds=2)
    mae = {r["model"]: r["mae"] for r in m["depth"]}
    assert mae["Tracker (final)"] < mae["Mileage baseline"]
    for f in ["predictions.csv.gz", "metrics.json", "gauges.csv", "tires.csv", "examples.npz", "gbm_final.joblib"]:
        assert (tmp_path / f).exists()
    json.loads((tmp_path / "metrics.json").read_text())
