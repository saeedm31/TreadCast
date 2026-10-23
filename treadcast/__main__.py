import argparse
import json

from .pipeline import run

p = argparse.ArgumentParser(prog="treadcast", description="Simulate the fleet, train the models, write artifacts.")
p.add_argument("--out", default="artifacts")
p.add_argument("--fast", action="store_true", help="small quick run (200 days, 3 epochs)")
a = p.parse_args()
m = run(a.out, days=200, epochs=3, folds=3) if a.fast else run(a.out)
print(json.dumps({r["model"]: round(r["mae"], 3) for r in m["depth"]}, indent=1))
