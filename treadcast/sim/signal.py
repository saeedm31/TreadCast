"""Radial acceleration seen by a sensor glued to the tire's inner liner.

Each wheel turn the sensor reads about v²/R (the centrifugal level, tens of g) while it
is off the road, drops to about 0 g while it crosses the contact patch, and shows a
spike when it enters and leaves the patch. After leaving, the tread blocks ring briefly.

The sensor does not send raw data. Like a real in-tire sensor it works on the edge:
it lines up many steady wheel turns by angle and sends the median "template" of the
±45° around the contact patch, plus the turn count used for the rolling radius.

How wear changes the signal (all small, all mixed with operating conditions):
- thinner tread -> smaller free radius -> more turns per km, higher centrifugal level
- thinner tread -> stiffer -> less deflection -> shorter contact patch
- thinner tread -> less rubber to smooth the edges -> sharper, taller entry/exit spikes
- thinner tread -> stiffer tread blocks -> higher ringing frequency
Pressure, load, temperature and speed move the same quantities, often by more.
"""

from __future__ import annotations

import numpy as np

from .. import config as C

THETA = np.linspace(-C.WINDOW_DEG, C.WINDOW_DEG, C.TEMPLATE_POINTS)


def tire_geometry(depth, new_depth, pressure_bar, wheel_load_kg, stiffness):
    """Free radius, deflection, rolling radius and contact patch half-angle (all arrays)."""
    depth = np.asarray(depth, float)
    worn = 1.0 - depth / new_depth
    r_free = C.CARCASS_RADIUS_MM + depth + 0.4 * (pressure_bar - 8.0)
    k_z = (330.0 + 85.0 * pressure_bar) * stiffness * (1.0 + 0.15 * worn)       # N/mm
    deflection = wheel_load_kg * C.G / k_z                                     # mm
    r_roll = r_free - deflection / 3.0
    half_patch = np.sqrt(np.maximum(2 * r_free * deflection - deflection ** 2, 1.0))
    phi_deg = np.degrees(half_patch / r_free)
    return r_free, deflection, r_roll, phi_deg


def ring_frequency(depth, new_depth, ring_hz, tire_temp_c, pressure_bar):
    return (ring_hz * (new_depth / np.maximum(depth, 1.5)) ** 0.25
            * (1.0 - 0.004 * (tire_temp_c - 45.0)) * (1.0 + 0.02 * (pressure_bar - 8.5)))


def clean_template(depth, new_depth, speed_kmh, wheel_load_kg, pressure_bar, tire_temp_c,
                   ring_hz, stiffness, mount=1.0):
    """Noise-free template, shape (n, TEMPLATE_POINTS), in g. Inputs are arrays of length n."""
    a = [np.atleast_1d(np.asarray(x, float)) for x in
         (depth, new_depth, speed_kmh, wheel_load_kg, pressure_bar, tire_temp_c, ring_hz, stiffness, mount)]
    depth, new_depth, speed, load, p, temp, ring_hz, stiffness, mount = np.broadcast_arrays(*a)
    _, deflection, r_roll, phi = tire_geometry(depth, new_depth, p, load, stiffness)
    v = speed / 3.6
    a_c = v ** 2 / (r_roll / 1000.0) / C.G                                     # centrifugal level, g
    rel = (depth / new_depth)[:, None]
    th = THETA[None, :]
    phi_ = phi[:, None]

    edge = 0.6 + 1.0 * rel                                                      # deg; thick tread = softer edge
    step = 1.0 - 0.5 * (np.tanh((th + phi_) / edge) - np.tanh((th - phi_) / edge))
    width = 1.0 + 1.2 * rel
    # spike height: mostly deflection (load, pressure) and how the sensor sits; wear adds a little
    amp_in = ((0.22 + 0.06 * (1 - rel) + 0.15 * (deflection[:, None] / 30.0 - 1.0))
              * mount[:, None] * (1.0 - 0.004 * (temp[:, None] - 45.0)) * (1.0 + 0.002 * (speed[:, None] - 65.0)))
    amp_out = amp_in * (0.97 - 0.08 * rel - 0.003 * (temp[:, None] - 45.0))
    peaks = (amp_in * np.exp(-0.5 * ((th + phi_ + 1.2 * width) / width) ** 2)
             + amp_out * np.exp(-0.5 * ((th - phi_ - 1.2 * width) / width) ** 2))

    f = ring_frequency(depth, new_depth, ring_hz, temp, p)[:, None]
    after = np.clip(th - phi_ - 1.2 * width, 0, None)
    t_ms = np.radians(after) * (r_roll[:, None] / 1000.0) / v[:, None] * 1000.0
    ring = 0.06 * np.exp(-t_ms / 3.0) * np.sin(2 * np.pi * f * t_ms / 1000.0) * (after > 0)

    return a_c[:, None] * (step + peaks + ring), a_c, r_roll


def sensor_template(rows, tires, rng):
    """What the sensor sends each driving day: the median template (with road and
    sensor noise left after averaging), the centrifugal level and the turn count."""
    t = tires.set_index("tire_id").loc[rows["tire_id"]]
    clean, a_c, r_roll = clean_template(
        rows["depth_true"].to_numpy(), rows["new_depth_mm"].to_numpy(), rows["speed_kmh"].to_numpy(),
        rows["wheel_load_kg"].to_numpy(), rows["pressure_bar"].to_numpy(), rows["tire_temp_c"].to_numpy(),
        t["ring_hz"].to_numpy(), t["stiffness"].to_numpy(), t["mount"].to_numpy())
    n = len(rows)
    n_revs = np.clip(rows["km_day"].to_numpy() * 2.0, 150, 2500)              # steady turns kept
    road = 0.10 * rows["roughness"].to_numpy() * a_c * 1.25 / np.sqrt(n_revs)
    noise = rng.normal(0, 1, (n, C.TEMPLATE_POINTS)) * (road[:, None] + C.SENSOR_NOISE_G / np.sqrt(n_revs)[:, None] + 0.15)
    template = (clean + noise).astype(np.float32)

    turns = rows["km_day"].to_numpy() * 1e6 / (2 * np.pi * r_roll)             # exact turn count
    return template, a_c, turns, n_revs
