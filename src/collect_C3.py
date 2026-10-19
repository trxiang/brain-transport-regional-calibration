"""Step C3 tables: collect the time-window (C3a_*) and inflow-transfer (C3b_*) cases written by run_case.py.

Writes results/stepC3a_timewindow.csv and results/stepC3b_transfer.csv (percentages for errors and drifts).
"""
import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
CASES = RES / "cases"

rows = []
for f in sorted(CASES.glob("C3a_*.json")):
    r = json.load(open(f))
    rows.append({"truth": r["truth"],
                 **{f"{w}_sD": r[w]["s_D_fit"] for w in ("full", "early", "late")},
                 **{f"{w}_r": r[w]["r_fit"] for w in ("full", "early", "late")},
                 "early_fit_roi_%": 100 * r["early"]["roi_L2"],
                 "early_fit_late_roi_%": 100 * r["early"]["late_roi_L2"],
                 "full_fit_late_roi_%": 100 * r["full"]["late_roi_L2"]})
pd.DataFrame(rows).sort_values("truth").to_csv(RES / "stepC3a_timewindow.csv", index=False)

rows = []
for f in sorted(CASES.glob("C3b_*.json")):
    r = json.load(open(f))
    if r.get("level", 0):
        continue
    rows.append({"truth": r["truth"], "inflow": r["inflow"],
                 "sD_std": r["s_D_fit_standard"], "sD_new": r["s_D_refit_new"],
                 "drift_sD_%": 100 * (r["s_D_refit_new"] / r["s_D_fit_standard"] - 1),
                 "drift_r_%": 100 * (r["r_refit_new"] / r["r_fit_standard"] - 1),
                 "transfer_roi_%": 100 * r["transfer_roi_L2"], "refit_roi_%": 100 * r["roi_L2_refit_new"],
                 "transfer_native_%": 100 * r["transfer_native_L2"], "refit_native_%": 100 * r["native_L2_refit_new"]})
pd.DataFrame(rows).sort_values(["truth", "inflow"]).to_csv(RES / "stepC3b_transfer.csv", index=False)
print("wrote", RES / "stepC3a_timewindow.csv", "and", RES / "stepC3b_transfer.csv")
