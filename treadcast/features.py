"""From the sensor's daily template plus vehicle data to one feature row per tire per day.

Feature groups (used for the ablation in the report):
- physics: rolling radius from turns vs GPS distance, radius from the centrifugal level,
  contact patch width
- signal: entry/exit spike height and width, edge sharpness, ringing frequency and energy
- context: speed, pressure, temperature, load, mileage, axle, tire model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .sim.signal import THETA, sensor_template

DTHETA = THETA[1] - THETA[0]

PHYSICS = ["r_roll_mm", "r_from_ac_mm", "patch_deg"]
SIGNAL = ["peak_in", "peak_out", "peak_ratio", "peak_width_deg", "edge_slope", "ring_hz", "ring_energy"]
CONTEXT = ["speed_kmh_meas", "pressure_bar", "tire_temp_c", "ambient_c", "load_est_kg",
           "km_since_install", "days_since_install", "new_depth_mm", "is_steer", "sku_code"]
ALL_FEATURES = PHYSICS + SIGNAL + CONTEXT
SKU_CODES = {s.name: i for i, s in enumerate([s for v in C.SKUS.values() for s in v])}


def template_features(norm: np.ndarray, speed_kmh: np.ndarray, r_mm: np.ndarray) -> pd.DataFrame:
    """Shape features from templates normalised by their centrifugal level (1 = off the road)."""
    n = len(norm)
    left, right = THETA < 0, THETA > 0
    below = norm < 0.5
    patch_deg = below.sum(axis=1) * DTHETA

    peak_in = norm[:, left].max(axis=1) - 1.0
    peak_out = norm[:, right].max(axis=1) - 1.0
    i_in = np.argmax(np.where(left, norm, -np.inf), axis=1)
    i_out = np.argmax(np.where(right, norm, -np.inf), axis=1)
    above_half = (norm - 1.0) > (peak_in[:, None] / 2.0)
    width = np.array([_run_length(above_half[k], i_in[k]) for k in range(n)]) * DTHETA
    slope = np.abs(np.diff(norm, axis=1)).max(axis=1) / DTHETA

    # ringing after the exit spike: FFT of the tail, angle converted to time with speed and radius
    ring_hz, ring_energy = np.zeros(n), np.zeros(n)
    for k in range(n):
        dt = np.radians(DTHETA) * (r_mm[k] / 1000.0) / (speed_kmh[k] / 3.6)       # s per sample
        start = i_out[k] + 2
        stop = min(start + int(0.012 / dt), len(THETA))                            # 12 ms after the spike
        seg = norm[k, max(start - 6, 0):min(stop + 6, len(THETA))] - 1.0
        if stop - start < 16:
            continue
        smooth = np.convolve(seg, np.ones(9) / 9.0, mode="same")                   # remove slow shape
        tail = (seg - smooth)[6:6 + stop - start]
        spec = np.abs(np.fft.rfft(tail * np.hanning(len(tail)), n=4096))
        freqs = np.fft.rfftfreq(4096, dt)
        band = (freqs > 250) & (freqs < 1600)
        j = np.argmax(np.where(band, spec, 0))
        ring_hz[k] = freqs[j]
        ring_energy[k] = float(np.sum(tail[: int(8.0 / DTHETA)] ** 2))
    return pd.DataFrame(dict(patch_deg=patch_deg, peak_in=peak_in, peak_out=peak_out,
                             peak_ratio=peak_out / np.maximum(peak_in, 1e-3), peak_width_deg=width,
                             edge_slope=slope, ring_hz=ring_hz, ring_energy=ring_energy))


def _run_length(mask_row, i):
    """Length of the run of True values around index i."""
    if not mask_row[i]:
        return 0
    lo = i
    while lo > 0 and mask_row[lo - 1]:
        lo -= 1
    hi = i
    while hi < len(mask_row) - 1 and mask_row[hi + 1]:
        hi += 1
    return hi - lo + 1


def build_dataset(rows: pd.DataFrame, tires: pd.DataFrame, seed: int = C.SEED):
    """Simulate the sensor for every driving day and compute features.
    Returns (table, templates): templates are normalised by the centrifugal level."""
    rng = np.random.default_rng(seed + 1)
    rows = rows.reset_index(drop=True)
    template, _, turns, n_revs = sensor_template(rows, tires, rng)

    out = rows.copy()
    out["speed_kmh_meas"] = rows["speed_kmh"] * (1 + rng.normal(0, 0.004, len(rows)))
    out["load_est_kg"] = rows["wheel_load_kg"] * (1 + rng.normal(0, C.LOAD_SENSOR_NOISE, len(rows)))
    out["n_revs"] = n_revs
    out["r_roll_mm"] = rows["gps_km"] * 1e6 / (2 * np.pi * turns)          # NaN when GPS is missing
    ac_level = np.median(np.concatenate([template[:, :24], template[:, -24:]], axis=1), axis=1)
    out["ac_level_g"] = ac_level
    v = out["speed_kmh_meas"].to_numpy() / 3.6
    out["r_from_ac_mm"] = v ** 2 / (ac_level * C.G) * 1000.0
    out["is_steer"] = (rows["axle"] == "steer").astype(int)
    out["sku_code"] = rows["sku"].map(SKU_CODES)

    norm = template / ac_level[:, None]
    r_for_time = out["r_roll_mm"].fillna(out["r_from_ac_mm"]).to_numpy()
    feats = template_features(norm, out["speed_kmh_meas"].to_numpy(), r_for_time)
    out = pd.concat([out, feats], axis=1)
    return out, norm.astype(np.float32)
