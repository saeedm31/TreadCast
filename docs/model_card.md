# Model card: tread depth and km left

| | |
|---|---|
| **Task** | Estimate the tread depth of each truck tire every day, and the km left until it reaches 3 mm |
| **Input** | Daily median template of the in-tire sensor's radial acceleration, pressure, temperature, GPS distance, load, speed |
| **Output** | Depth with a 95% range, wear rate, km left and safe km left, status (replace now / plan a change / OK) |
| **Data** | Simulated fleet (see [data card](data_card.md)) |
| **Owner** | Saeed Hojjatpanah |

## Models

| Model | Algorithm | Notes |
|---|---|---|
| Baseline | mileage only | the rule to beat |
| 1D CNN | PyTorch: 4 convolution layers on the 256-point template, plus the operating conditions | mileage left out on purpose; falls back to PCA of the template + gradient boosting when PyTorch is missing |
| Daily depth | scikit-learn `HistGradientBoostingRegressor` (400 trees) | fuses physics, signal and context features and the CNN's out-of-fold estimate; monotone: more mileage never means more tread |
| Tracker | Kalman filter per tire, state = depth and wear rate | depth only goes down; km left = (depth - limit) / wear rate; the safe value uses the low end of the range |

## Intended use
Planning tire changes for a truck fleet: which tires to check or replace soon. A person decides; the estimate does
not replace a legal inspection.

## Not intended for
Proving that a tire is legal, or any decision without a check at the next service. Not validated on real tires.

## Evaluation
Three whole trucks, one per route type, are never trained on; training only uses days before a cutoff. Scored every
day against the true depth, against real gauge readings, and forward in time after the cutoff. Also: km-left error,
how often the safe km left is too high, and the share of below-limit days each policy flagged in time. The numbers
are in the [README](../README.md#-results) and in `artifacts/metrics.json`, rebuilt by GitHub Actions on every change.

## Known limits
- Simulated physics: real sensors, mountings and tire builds vary more.
- The true km left uses each tire's lifetime wear rate, which a real fleet only knows after the tire is removed.
- Urban trucks at low speed are the hardest case: the tread ringing is harder to read.

## Monitoring (production plan)
At every gauge reading: error by route, axle and tire model. Share of tires flagged late. Retrain when the error at
readings rises, or when a new tire model or route type appears.
