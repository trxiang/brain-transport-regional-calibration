"""Collect the three-level mesh study for T1, T2, T3, T3wide and E5 and extrapolate to zero mesh size.

Richardson from three levels with refinement ratio 2: rho = (q1 - q0) / (q2 - q1), observed order log2(rho),
limit q2 + (q2 - q1) / (rho - 1). When rho <= 1 (no monotone convergence) the limit is reported as q2 and the
order as NaN. The limiting contrast is the ratio of the limiting native and regional errors (not extrapolated
separately). T2 levels 1-2 come from Step A (stepA_L1_standard.json, stepA_L2_standard.json).
Writes results/convergence_summary.csv and results/convergence_summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
CONV = RES / "convergence"
KEYS = ["s_D_fit", "r_fit", "roi_L2", "native_L2", "hidden_fraction", "contrast_native_over_roi"]


def load(case, level):
    if case == "T2" and level > 0:
        return json.load(open(RES / f"stepA_L{level}_standard.json"))
    f = CONV / f"{case}_L{level}.json"
    return json.load(open(f)) if f.exists() else None


def richardson(q):
    q0, q1, q2 = q
    if q2 == q1:
        return q2, np.nan
    rho = (q1 - q0) / (q2 - q1)
    if rho <= 1:
        return q2, np.nan
    return q2 + (q2 - q1) / (rho - 1), float(np.log2(rho))


rows, summary = [], {}
for case in ("T1", "T2", "T3", "T3wide", "E5"):
    levels = [load(case, lv) for lv in (0, 1, 2)]
    for lv, d in enumerate(levels):
        if d is None:
            continue
        rows.append({"case": case, "level": lv, "nodes": d.get("nodes"),
                     **{k: d.get(k) for k in KEYS}, "negmass_truth": d.get("negmass_truth"),
                     "masked_1pct_rule": d.get("masked")})
    if all(d is not None and d.get("roi_L2") is not None for d in levels):
        ext = {}
        for k in KEYS:
            v, p = richardson([d[k] for d in levels])
            ext[k] = {"extrapolated": v, "observed_order": p}
        ext["contrast_native_over_roi"] = {"extrapolated": ext["native_L2"]["extrapolated"] / ext["roi_L2"]["extrapolated"],
                                           "observed_order": float("nan")}
        ext["masked_1pct_rule"] = bool(ext["roi_L2"]["extrapolated"] <= 0.01 and ext["native_L2"]["extrapolated"] > 0.01
                                       and ext["hidden_fraction"]["extrapolated"] >= 0.8)
        summary[case] = ext
        rows.append({"case": case, "level": "limit", **{k: ext[k]["extrapolated"] for k in KEYS},
                     "masked_1pct_rule": ext["masked_1pct_rule"]})
pd.DataFrame(rows).to_csv(RES / "convergence_summary.csv", index=False)
json.dump(summary, open(RES / "convergence_summary.json", "w"), indent=2)
print(pd.DataFrame(rows).to_string())
