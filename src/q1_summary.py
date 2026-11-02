"""Collect results/nd/q1/*.json into results/nd/q1_summary.csv and a Markdown table."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import twin  # noqa: E402

LABELS = {
    "A0": "Perfect analyst (true field, true CSF input, 4 sessions, tissue curves)",
    "A1": "CSF input linearly interpolated between sessions",
    "A2": "CSF input amplitude bias",
    "A3": "CSF partial volume in tissue values",
    "A4": "Node noise sigma 0.02 mmol/L",
    "A5": "Sessions used",
    "A6": "Clearance fixed at zero",
    "A7": "Full-field objective",
    "B1": "Literature (tortuosity) diffusion field",
    "B2": "DTI isotropic (no anisotropy) field",
    "C1": "Combined realistic errors",
    "D": "Other truths (no clearance; genuine alpha 3.5)",
    "E": "Truth with tissue flow, analyst without flow",
}

rows = []
for f in sorted((nd.ND_OUT / "q1").glob("*.json")):
    d = json.load(open(f))
    a = d["analyst"]
    rows.append({"case": d["case"], "block": d["case"].split("_")[0], "truth": d["truth"],
                 "alpha_true": d["alpha_true"], "alpha_fit": d["alpha_fit"],
                 "apparent_enhancement_pct": 100 * d["apparent_enhancement"],
                 "r_true": d["r_true"], "r_fit": d["r_fit"], "roi_L2_pct": 100 * d["roi_L2"],
                 "native_L2_pct": 100 * d["native_L2"], "hidden_pct": 100 * d["hidden_fraction"],
                 "active_bound": d["fit"]["active_bound"], "field": a["field"], "bc": a["bc"], "amp": a["amp"],
                 "obs": a["obs"], "window": a["window"], "pv": a["pv"], "noise": a["noise"], "fix_r": a["fix_r"]})
df = pd.DataFrame(rows)
df.to_csv(nd.ND_OUT / "q1_summary.csv", index=False)
cols = ["case", "alpha_true", "alpha_fit", "apparent_enhancement_pct", "r_fit", "roi_L2_pct", "native_L2_pct",
        "active_bound"]
with open(nd.ND_OUT / "q1_summary.md", "w") as fh:
    for b, lab in LABELS.items():
        sub = df[df.block == b] if b != "D" else df[df.block == "D"]
        if len(sub):
            fh.write(f"\n### {b}: {lab}\n\n" + sub[cols].round(4).to_markdown(index=False) + "\n")
print(df[cols].round(4).to_string(index=False))
