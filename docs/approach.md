# Approach in detail

## 1. What the sensor sees

A sensor glued to the inner liner turns with the tire. Its radial axis points to the wheel centre.

- **Off the road** it feels the centrifugal acceleration a = v²/R: about 95 g at 80 km/h on a 315/80 R22.5 tire.
- **In the contact patch** the liner is flat on the road, so the radial acceleration drops to about 0 g.
- **At entry and exit** the liner bends sharply, which gives a spike above the off-road level.
- **After exit** the tread blocks snap back and ring for a few milliseconds.

How wear changes this, and what else changes the same thing:

| Quantity | Effect of wear (less tread) | Same quantity also moved by |
|---|---|---|
| Rolling radius (turns per km) | smaller radius, more turns | load (deflection), pressure, GPS error |
| Centrifugal level v²/R | higher | speed (squared), load |
| Contact patch length | shorter (worn tread is stiffer) | load (strongly), pressure |
| Entry/exit spikes | taller and sharper | load, pressure, temperature, sensor mounting |
| Tread ringing frequency | higher | temperature, pressure, tire model, build spread |

So the operating conditions are not noise to remove. They go into the model as features, so it can tell a worn
tire apart from a heavy, cold or under-inflated day.

## 2. Edge processing

Sending raw waveforms from inside a tire is not realistic: battery and radio limit it. The sensor (simulated)
does three things on board:

1. keeps only steady driving (constant speed, straight line, warm tire),
2. resamples each wheel turn to angle (so speed changes don't smear the shape),
3. takes the median over hundreds to thousands of turns. Road bumps are random from turn to turn and average out;
   the wear signature repeats every turn and stays.

It sends the ±45° around the contact patch (256 points) once a day, plus the day's turn count.

## 3. Simulator

`treadcast/sim/` builds a 12-truck fleet (6 tires each: 2 steer, 4 drive) over two years:

- **Routes:** long-haul (≈640 km/day, 80 km/h), regional and urban (more stop-and-go, rougher roads, faster wear).
- **Each day:** km, load, steady speed, ambient temperature by season, tire temperature, pressure
  (slow natural loss, occasional slow punctures, refills at alarms and services).
- **True wear** per km depends on axle, tire model, route, load, under-inflation, temperature, a per-tire factor
  (alignment, compound batch) and the depth itself.
- **Services** every 30,000 km or 75 days: a technician gauges each tire (±0.3 mm) and replaces tires at 4 mm or less.
- **Signal:** each tire-day's template is generated from the tire's true geometry and conditions, with road and
  sensor noise scaled by 1/√(turns averaged). Each tire also has small unknown build differences (stiffness,
  ringing frequency, sensor mounting), so no single feature reads off the depth.

## 4. Labels

| Source | When | Weight |
|---|---|---|
| New tire depth (spec sheet) | at fitting | 1.0 |
| Gauge reading | each service | 1.0 |
| Straight-line interpolation over km between two known points | days in between | 0.3 |

No label is made after the last known point (no extrapolation). Pseudo-labels use a later measurement, so they
are only used for training.

## 5. Models

- **Mileage baseline:** new depth − average wear rate per axle × km. The model to beat.
- **1D CNN (PyTorch):** 4 convolution blocks over the 256-point template, joined with speed, pressure, temperature,
  load, axle and new depth. Mileage is left out, so the CNN has to read the shape. Trained with weighted MSE.
  Its predictions for the training rows are **out-of-fold** (grouped by tire), so stacking does not leak.
- **Gradient boosting** (scikit-learn `HistGradientBoostingRegressor`, the same family as XGBoost/LightGBM):
  physics + signal + context + CNN output. Handles missing GPS days natively; monotone constraint so mileage and
  age can only lower the estimate. An ablation adds one feature group at a time.
- **Per-tire tracker:** Kalman filter with state (depth, wear rate). Prediction: depth −= rate × km. Update: the
  day's model estimate, weighted by the model's out-of-fold noise at real readings. Starts at the known new depth and
  the fleet's average rate. Output: depth (never goes up), rate, and their uncertainty.
- **Remaining km** = (depth − 3 mm) ÷ rate. **Safe km** = (depth − 1.64σ − 3 mm) ÷ (rate + 1.64σ): it errs towards
  changing early, because over-estimating remaining life is the costly mistake.

## 6. Evaluation

- **Held out by tire:** three whole trucks (one per route) are never used for training.
- **Forward in time:** training only uses data (and gauge readings) from before day 500.
- **Metrics:** MAE, RMSE, R², bias against the true depth on every tire-day; MAE at real gauge readings; by axle,
  route, season and speed; km-left error; how often the safe km is too high; the share of below-limit days each
  policy had flagged (TreadCast vs service checks only).

## 7. From demo to production

| Step | Demo | Production |
|---|---|---|
| Data | simulator | sensor platform stream + vehicle telematics + service records |
| Features | `features.py` | the same code in a feature pipeline, one definition for training and scoring |
| Training | `python -m treadcast` / `train` image | scheduled job, data and model versions tracked (e.g. MLflow) |
| Serving | Streamlit app / `app` image | nightly batch scoring to the fleet portal; same container on ECS/Kubernetes |
| Monitoring | model report | drift per tire model and season, error at every new gauge reading |
| Feedback | - | every gauge reading at a service becomes a new label |

Gates before trusting it on a fleet: depth MAE under 1 mm on held-out tires, stable across tire models, axle
positions, seasons and speeds, and a low rate of over-estimated remaining km.
