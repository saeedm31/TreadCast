"""TreadCast: tread depth and remaining km from an in-tire sensor (simulated truck fleet)."""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from treadcast import config as C
from treadcast.sim.signal import THETA, clean_template, ring_frequency, tire_geometry
from treadcast.tracker import Z

ART = os.getenv("TREADCAST_ARTIFACTS", "artifacts")
PLAN_KM = 15000                       # "plan a change" when fewer km are left than this

st.set_page_config(page_title="TreadCast", page_icon="🛞", layout="wide")

BLUE, ORANGE, GREY, GREEN, RED = "#2F5BB7", "#D9692A", "#8A94A6", "#2E8B57", "#C0392B"


@st.cache_data(show_spinner=False)
def load():
    p = pd.read_csv(os.path.join(ART, "predictions.csv.gz"), parse_dates=["date"])
    g = pd.read_csv(os.path.join(ART, "gauges.csv"))
    t = pd.read_csv(os.path.join(ART, "tires.csv"))
    with open(os.path.join(ART, "metrics.json")) as f:
        m = json.load(f)
    ex = dict(np.load(os.path.join(ART, "examples.npz")))
    return p, g, t, m, ex


if not os.path.exists(os.path.join(ART, "predictions.csv.gz")):
    st.error("No model results yet. Run `python -m treadcast` (or `docker compose run pipeline`) first.")
    st.stop()

pred, gauges, tires, metrics, examples = load()
start = pd.Timestamp(C.START_DATE)

# --- sidebar ---
with st.sidebar:
    st.title("🛞 TreadCast")
    st.caption("Tread depth and km left for every truck tire, from an in-tire sensor.")
    st.warning("**Simulated data.** A physics-based simulator stands in for a real fleet, so the "
               "true depth is known and the models can be scored honestly.", icon="🧪")
    last_day = int(pred["day"].max())
    day = st.slider("Fleet date (day of the simulation)", 0, last_day, last_day)
    st.caption(f"{(start + pd.Timedelta(days=day)).date():%d %b %Y}")
    st.caption(f"Plan a change below **{C.REPLACE_AT_MM:g} mm** (legal limit {C.LEGAL_LIMIT_MM} mm). "
               f"Trucks {', '.join(C.TEST_TRUCKS)} were never used for training.")

fleet_tab, tire_tab, signal_tab, report_tab, how_tab = st.tabs(
    ["🚛 Fleet", "🛞 One tire", "📡 Sensor signal", "📊 Model report", "ℹ️ How it works"])


def status(row):
    if row["remaining_km_safe"] <= 0:
        return "🔴 Replace now"
    if row["remaining_km_safe"] < PLAN_KM:
        return "🟠 Plan a change"
    return "🟢 OK"


# --- fleet ---
with fleet_tab:
    now = pred[pred["day"] <= day].sort_values("day").groupby(["truck", "position"]).tail(1)
    now = now[now["day"] >= day - 14]                      # tires still on the road
    now = now.assign(status=now.apply(status, axis=1))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tires tracked", len(now))
    c2.metric("🔴 Replace now", int((now["status"] == "🔴 Replace now").sum()))
    c3.metric("🟠 Plan a change", int((now["status"] == "🟠 Plan a change").sum()))
    c4.metric("🟢 OK", int((now["status"] == "🟢 OK").sum()))
    view = pd.DataFrame({
        "Status": now["status"],
        "Truck": now["truck"] + np.where(now["split"] == "test", " ★", ""),
        "Route": now["route"],
        "Position": now["position"],
        "Tire model": now["sku"],
        "Depth (mm)": now["depth_kf"].round(1).astype(str) + " ± " + (Z * now["depth_sd"]).round(1).astype(str),
        "Km left (safe)": now["remaining_km_safe"].round(-2).astype(int),
        "Days left (approx.)": (now["remaining_km_safe"] / now["km_per_day_30"].clip(lower=50)).round().astype(int),
        "Wear (mm / 1000 km)": now["rate_kf"].round(3),
        "True depth (sim)": now["depth_true"].round(1),
    })
    order = {"🔴 Replace now": 0, "🟠 Plan a change": 1, "🟢 OK": 2}
    view = view.sort_values(["Status", "Km left (safe)"], key=lambda s: s.map(order) if s.name == "Status" else s)
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption("★ = test truck, never seen in training. *Km left (safe)* uses the low end of the depth range and "
               "the high end of the wear rate, so it errs towards changing early. *True depth* is only known "
               "because the data is simulated.")

# --- one tire ---
with tire_tab:
    c1, c2, c3 = st.columns(3)
    truck = c1.selectbox("Truck", sorted(pred["truck"].unique()), index=sorted(pred["truck"].unique()).index(C.TEST_TRUCKS[0]))
    position = c2.selectbox("Position", [p.name for p in C.POSITIONS])
    lives = sorted(pred.loc[(pred["truck"] == truck) & (pred["position"] == position), "tire_id"].unique(),
                   key=lambda s: int(s.rsplit("-", 1)[1]))
    tire_id = c3.selectbox("Tire (1 = first fitted)", lives)
    d = pred[pred["tire_id"] == tire_id].sort_values("day")
    g = gauges[(gauges["tire_id"] == tire_id) & (~gauges["is_new"])]
    last = d.iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Depth now (estimate)", f"{last['depth_kf']:.1f} mm", f"true {last['depth_true']:.1f} mm (sim)", delta_color="off")
    m2.metric("Km left (safe)", f"{last['remaining_km_safe']:,.0f} km", f"expected {last['remaining_km']:,.0f} km", delta_color="off")
    m3.metric("Wear rate", f"{last['rate_kf']:.3f} mm/1000 km")
    m4.metric("Error vs truth (this tire)", f"{np.mean(np.abs(d['depth_kf'] - d['depth_true'])):.2f} mm",
              f"mileage only: {np.mean(np.abs(d['pred_baseline'] - d['depth_true'])):.2f} mm", delta_color="off")

    x = d["km_since_install"] / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([d["depth_kf"] + Z * d["depth_sd"], (d["depth_kf"] - Z * d["depth_sd"])[::-1]]),
                             fill="toself", fillcolor="rgba(47,91,183,0.15)", line=dict(width=0), name="Tracker range (95%)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=d["pred_final"], mode="markers", marker=dict(size=4, color=GREY, opacity=0.5),
                             name="Daily model estimate"))
    fig.add_trace(go.Scatter(x=x, y=d["pred_baseline"], line=dict(color=ORANGE, dash="dot"), name="Mileage only"))
    fig.add_trace(go.Scatter(x=x, y=d["depth_kf"], line=dict(color=BLUE, width=3), name="TreadCast (tracker)"))
    fig.add_trace(go.Scatter(x=x, y=d["depth_true"], line=dict(color="#14213D", dash="dash"), name="True depth (sim)"))
    if len(g):
        fig.add_trace(go.Scatter(x=g["km_since_install"] / 1000, y=g["depth_gauge"], mode="markers",
                                 marker=dict(size=11, symbol="diamond", color=GREEN), name="Gauge reading at service"))
    fig.add_hline(y=C.REPLACE_AT_MM, line=dict(color=RED, dash="dash"), annotation_text="plan change")
    fig.add_hline(y=C.LEGAL_LIMIT_MM, line=dict(color=RED), annotation_text="legal limit")
    fig.update_layout(height=430, xaxis_title="km since fitted (thousands)", yaxis_title="tread depth (mm)",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=d["remaining_true_km"] / 1000, line=dict(color="#14213D", dash="dash"), name="True km left (sim)"))
    fig2.add_trace(go.Scatter(x=x, y=d["remaining_km"] / 1000, line=dict(color=BLUE), name="Expected km left"))
    fig2.add_trace(go.Scatter(x=x, y=d["remaining_km_safe"] / 1000, line=dict(color=ORANGE), name="Safe km left"))
    fig2.update_layout(height=320, xaxis_title="km since fitted (thousands)", yaxis_title="km left (thousands)",
                       legend=dict(orientation="h", y=-0.3), margin=dict(t=20))
    st.plotly_chart(fig2, width="stretch")

# --- sensor signal ---
with signal_tab:
    st.markdown("The sensor sits on the inner liner. Each wheel turn it reads the **centrifugal level** (v²/R), "
                "drops to **about 0 g** while crossing the contact patch, and spikes at the **entry and exit**. "
                "Move the sliders: wear changes the signal only a little, while load, pressure and speed change it more. "
                "That is why the model needs the operating conditions as inputs.")
    c1, c2 = st.columns([1, 2])
    with c1:
        axle = st.radio("Axle", ["drive", "steer"], horizontal=True)
        pos = next(p for p in C.POSITIONS if p.axle == axle)
        depth = st.slider("Tread depth (mm)", 2.0, float(pos.new_depth_mm), 8.0, 0.5)
        speed = st.slider("Speed (km/h)", 30, 95, 80)
        load = st.slider("Wheel load (kg)", int(pos.empty_load_kg), int(pos.full_load_kg), int(pos.full_load_kg))
        bar = st.slider("Pressure (bar, hot)", 7.0, 10.0, 8.8, 0.1)
        temp = st.slider("Tire temperature (°C)", 0, 70, 45)
        ring = C.SKUS[axle][0].ring_hz
    new_t, _, _ = clean_template(pos.new_depth_mm, pos.new_depth_mm, speed, load, bar, temp, ring, 1.0)
    cur_t, ac, rr = clean_template(depth, pos.new_depth_mm, speed, load, bar, temp, ring, 1.0)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=THETA, y=new_t[0], line=dict(color=GREY, dash="dash"), name=f"new ({pos.new_depth_mm:g} mm)"))
        fig.add_trace(go.Scatter(x=THETA, y=cur_t[0], line=dict(color=BLUE, width=3), name=f"{depth:g} mm"))
        fig.update_layout(height=380, xaxis_title="angle from contact patch centre (°)", yaxis_title="radial acceleration (g)",
                          legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        _, defl, r_roll, phi = tire_geometry(depth, pos.new_depth_mm, bar, load, 1.0)
        _, _, r_new, phi_new = tire_geometry(pos.new_depth_mm, pos.new_depth_mm, bar, load, 1.0)
        k1, k2, k3 = st.columns(3)
        k1.metric("Rolling radius", f"{float(r_roll[0] if np.ndim(r_roll) else r_roll):.1f} mm",
                  f"{float(np.ravel(r_roll)[0] - np.ravel(r_new)[0]):+.1f} mm vs new")
        k2.metric("Contact patch", f"{2 * float(np.ravel(phi)[0]):.1f}°", f"{2 * float(np.ravel(phi)[0] - np.ravel(phi_new)[0]):+.1f}° vs new")
        f_ring = float(np.ravel(ring_frequency(depth, pos.new_depth_mm, ring, temp, bar))[0])
        f_new = float(np.ravel(ring_frequency(pos.new_depth_mm, pos.new_depth_mm, ring, temp, bar))[0])
        k3.metric("Tread ringing", f"{f_ring:.0f} Hz", f"{f_ring - f_new:+.0f} Hz vs new")

    st.markdown("**Real (noisy) templates from a test tire** - what the sensor sends each day, the median of many wheel turns.")
    tid = str(examples["tire_id"][0])
    sel = examples["tire_id"].astype(str) == tid
    fig = go.Figure()
    for tpl, dep, dd in zip(examples["templates"][sel], examples["depth"][sel], examples["day"][sel]):
        fig.add_trace(go.Scatter(x=THETA, y=tpl.astype(float), name=f"day {int(dd)}: {dep:.1f} mm"))
    fig.update_layout(height=320, xaxis_title="angle (°)", yaxis_title="level (1 = off the road)",
                      legend=dict(orientation="h", y=-0.3), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

# --- model report ---
with report_tab:
    st.markdown(f"Scored on **{metrics['test']['tires']} tires** of trucks {', '.join(metrics['test']['trucks'])}, "
                f"which the models never saw ({metrics['test']['rows']:,} tire-days). Trained on "
                f"{metrics['train']['labelled_rows']:,} labelled tire-days from the other trucks, before day "
                f"{metrics['train']['cutoff_day']}, of which only {metrics['train']['real_readings']} are real gauge readings.")
    dep = pd.DataFrame(metrics["depth"])
    fig = go.Figure(go.Bar(x=dep["mae"], y=dep["model"], orientation="h", text=dep["mae"].round(2),
                           marker_color=[ORANGE] + [GREY] * (len(dep) - 2) + [BLUE]))
    fig.update_layout(height=340, xaxis_title="mean absolute error (mm), lower is better", yaxis=dict(autorange="reversed"),
                      margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Template model backend: **{metrics['backend']}**. Each GBM row adds one feature group: "
               "context (speed, pressure, temperature, load, mileage) → physics (rolling radius, contact patch) → "
               "signal (spikes, edges, ringing) → the CNN's reading of the raw template.")

    r, s = metrics["remaining"], metrics["safety"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Km left: typical error", f"{100 * r['median_rel_error']:.0f}%")
    k2.metric("Safe km left too high (days)", f"{100 * r['safe_over_estimate_rate']:.0f}%",
              help="Share of days where the safe km-left estimate was higher than the true km left.")
    k3.metric("Days below 3 mm flagged", f"{100 * s.get('tracker_flagged_share', 0):.0f}%",
              f"checks only: {100 * s.get('service_only_flagged_share', 0):.0f}%", delta_color="off",
              help="Days a tire was below 3 mm and TreadCast had flagged it, vs relying on service checks only.")
    gc = metrics.get("gauge_check", {})
    if gc:
        k4.metric("Error vs gauge readings", f"{gc['Tracker (final)']['mae']:.2f} mm",
                  f"mileage only: {gc['Mileage baseline']['mae']:.2f} mm", delta_color="off")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**By route**")
    c1.dataframe(pd.DataFrame(metrics["by_route"]), hide_index=True)
    c2.markdown("**By speed** (daily model)")
    c2.dataframe(pd.DataFrame(metrics["by_speed"]), hide_index=True)
    c3.markdown("**By axle**")
    c3.dataframe(pd.DataFrame(metrics["by_axle"]), hide_index=True)
    st.info("These numbers show the method works **on the simulation**. They are not a claim about real tires: "
            "a real project would validate on instrumented test-drum and fleet data first.", icon="ℹ️")

# --- how it works ---
with how_tab:
    st.markdown("""
**The problem.** Tread depth is never measured directly by the sensor, and a technician gauges it only a few times
per tire life. Fleets replace tires at services, so a tire that wears fast can run below the limit between two visits.

**The pipeline**
1. **Sensor (edge):** lines up thousands of steady wheel turns by angle and sends the median template of the
   contact-patch area, plus the turn count. Random road bumps average out; the wear signature stays.
2. **Features:** rolling radius from turns vs GPS distance, radius from the centrifugal level, contact patch width,
   entry/exit spikes, edge sharpness, tread ringing frequency, plus speed, pressure, temperature, load, mileage.
3. **1D CNN** reads the raw template shape. Its output is one more feature (out-of-fold, so no leakage).
4. **Gradient boosting** fuses everything into a daily depth estimate. Mileage may only push depth down.
5. **Per-tire tracker** (Kalman filter) turns the noisy daily estimates into one depth that only goes down,
   a wear rate, and a range. **Km left** = (depth − limit) ÷ wear rate; the *safe* value errs towards early.
6. **Labels:** gauge readings at services plus the known new depth; in between, straight-line pseudo-labels
   with a lower weight. Scores use only unseen trucks.

**Feedback loop:** every gauge reading at a service becomes a new label, so the model improves with every inspection.

Code, Docker setup and details: [github.com/saeedm31/TreadCast](https://github.com/saeedm31/TreadCast)
""")
