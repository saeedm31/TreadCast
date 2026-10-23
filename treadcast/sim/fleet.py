"""Simulated truck fleet: daily driving, operating conditions, true tread wear, services.

The simulator knows the true tread depth of every tire on every day. The models never
see it: they only get what a real system would get (sensor data, vehicle data and
gauge readings at services). The truth is used only to score the models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


def _trucks():
    out, n = [], 1
    for route in C.ROUTES:
        for _ in range(route.trucks):
            out.append((f"T{n:02d}", route))
            n += 1
    return out


class _Tire:
    """One physical tire, from fitting to removal."""

    def __init__(self, rng, truck, pos, n, day):
        self.id = f"{truck}-{pos.name}-{n}"
        self.truck, self.pos, self.install_day = truck, pos, day
        skus = C.SKUS[pos.axle]
        self.sku = skus[rng.integers(len(skus))]
        self.depth = pos.new_depth_mm
        self.km = 0.0
        # each tire wears a bit differently (compound batch, alignment, mounting)
        self.tire_factor = float(rng.lognormal(0.0, 0.18))
        if pos.axle == "steer" and rng.random() < 0.15:
            self.tire_factor *= 1.35            # misaligned axle
        # small build differences the models do not know about
        self.stiff_dev = float(rng.lognormal(0.0, 0.03))
        self.ring_dev = float(rng.lognormal(0.0, 0.04))
        self.mount = float(rng.lognormal(0.0, 0.10))     # how firmly the sensor sits on the liner
        self.cold_bar = pos.cold_pressure_bar
        self.leak = 0.003                       # bar per day, normal loss
        self.removed_day, self.reason = None, None


def simulate_fleet(seed: int = C.SEED, days: int = C.DAYS):
    """Returns (rows, gauges, tires): one row per tire per driving day, the gauge
    readings taken at services, and one row per tire life."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(C.START_DATE, periods=days, freq="D")
    rows, gauges, tires = [], [], []

    for truck, route in _trucks():
        fitted = {p.name: _Tire(rng, truck, p, 1, 0) for p in C.POSITIONS}
        count = {p.name: 1 for p in C.POSITIONS}
        km_since_service, last_service = 0.0, 0
        gps_bias = rng.normal(0, 0.0008)                 # this truck's GPS / odometer bias
        for t in fitted.values():
            gauges.append(_gauge(t, 0, exact=True))      # a new tire's depth is known

        for day, date in enumerate(dates):
            weekend = date.dayofweek >= 5
            drives = rng.random() < (route.work_prob * (0.25 if weekend else 1.0))
            ambient = 14.0 + 9.0 * np.sin(2 * np.pi * (date.dayofyear - 110) / 365.0) + rng.normal(0, 3)

            if drives:
                km = float(np.clip(rng.lognormal(np.log(route.km_mean), 0.25), 20, 1100))
                load = float(np.clip(rng.normal(route.load_mean, 0.22), 0.0, 1.0))
                speed = float(np.clip(rng.normal(route.speed_kmh, 6.0), 25, 95))
                roughness = float(route.roughness * rng.lognormal(0, 0.25))
                gps_ok = rng.random() > C.GPS_MISSING_PROB
                gps_err = gps_bias + rng.normal(0, C.GPS_DISTANCE_NOISE)
            for t in fitted.values():
                pos = t.pos
                # pressure: slow natural loss, sometimes a slow puncture
                if rng.random() < 0.002:
                    t.leak = rng.uniform(0.03, 0.08)
                t.cold_bar -= t.leak
                if t.cold_bar < pos.cold_pressure_bar - 1.0:      # pressure alarm -> refilled
                    t.cold_bar, t.leak = pos.cold_pressure_bar, 0.003
                if not drives:
                    continue
                wheel_load = pos.empty_load_kg + load * (pos.full_load_kg - pos.empty_load_kg)
                tire_temp = ambient + 15.0 + 10.0 * load + 0.1 * speed + rng.normal(0, 2)
                hot_bar = t.cold_bar * (tire_temp + 273.15) / 293.15
                under = max(0.0, (pos.cold_pressure_bar - t.cold_bar) / pos.cold_pressure_bar)
                rate = (pos.base_wear * t.sku.wear_factor * t.tire_factor * route.wear_factor
                        * (0.7 + 0.6 * load) * (1.0 + 1.5 * under) * (1.0 + 0.008 * (ambient - 15.0))
                        * (0.9 + 0.25 * t.depth / pos.new_depth_mm))
                wear = rate * km / 1000.0 * rng.lognormal(0, 0.05)
                t.depth = max(t.depth - wear, 0.3)
                t.km += km
                rows.append(dict(
                    day=day, date=date, truck=truck, route=route.name, position=pos.name, axle=pos.axle,
                    tire_id=t.id, sku=t.sku.name, new_depth_mm=pos.new_depth_mm,
                    km_day=km, km_since_install=t.km, days_since_install=day - t.install_day,
                    speed_kmh=speed, load_frac=load, wheel_load_kg=wheel_load, ambient_c=ambient,
                    tire_temp_c=tire_temp, pressure_bar=hot_bar, cold_pressure_bar=t.cold_bar,
                    roughness=roughness, gps_km=km * (1 + gps_err) if gps_ok else np.nan,
                    wear_rate_true=rate, depth_true=t.depth,
                ))
            if drives:
                km_since_service += km

            # service: gauge every tire, replace worn ones, refill pressure
            if km_since_service >= C.SERVICE_EVERY_KM or (day - last_service) >= C.SERVICE_EVERY_DAYS:
                km_since_service, last_service = 0.0, day
                for name, t in list(fitted.items()):
                    g = _gauge(t, day, rng=rng)
                    gauges.append(g)
                    t.cold_bar, t.leak = t.pos.cold_pressure_bar, 0.003
                    if g["depth_gauge"] <= C.SERVICE_REPLACE_MM:
                        t.removed_day, t.reason = day, "worn"
                        tires.append(_tire_row(t))
                        count[name] += 1
                        new = _Tire(rng, truck, t.pos, count[name], day + 1)
                        fitted[name] = new
                        gauges.append(_gauge(new, day + 1, exact=True))

        for t in fitted.values():
            t.reason = "in use"
            tires.append(_tire_row(t))

    rows = pd.DataFrame(rows)
    gauges = pd.DataFrame(gauges)
    tires = pd.DataFrame(tires)
    return rows, gauges, tires


def _gauge(t: _Tire, day: int, exact: bool = False, rng=None) -> dict:
    noise = 0.0 if exact else rng.normal(0, C.GAUGE_NOISE_MM)
    return dict(tire_id=t.id, truck=t.truck, position=t.pos.name, day=day, km_since_install=t.km,
                depth_gauge=round(t.depth + noise, 1), depth_true=t.depth, is_new=exact)


def _tire_row(t: _Tire) -> dict:
    return dict(tire_id=t.id, truck=t.truck, position=t.pos.name, axle=t.pos.axle, sku=t.sku.name,
                install_day=t.install_day, removed_day=t.removed_day, reason=t.reason,
                km=t.km, final_depth=t.depth, tire_factor=t.tire_factor, ring_hz=t.sku.ring_hz * t.ring_dev,
                stiffness=t.sku.stiffness * t.stiff_dev, mount=t.mount, new_depth_mm=t.pos.new_depth_mm)
