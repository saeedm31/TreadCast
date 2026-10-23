# artifacts/

Written by the pipeline (`python -m treadcast` or `docker compose run --rm pipeline`).
GitHub Actions rebuilds them on every change to the model code and commits the files the app reads:

| File | What it is |
|---|---|
| `metrics.json` | scores on the held-out trucks |
| `predictions.csv.gz` | one row per tire per driving day: true depth, every model's estimate, tracker output, km left |
| `gauges.csv` | gauge readings at services (and new-tire depths) |
| `tires.csv` | one row per tire life |
| `examples.npz` | a few real sensor templates per test tire, for the app |

Model files (`*.joblib`, `*.pt`) are not committed; CI keeps them as workflow artifacts.
