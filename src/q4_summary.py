"""Summarise Q4: real-data calibration under each analysis choice. Writes results/nd/q4_summary.csv/.md."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402

rows = []
for f in sorted((twin.REPO / "results/nd/q4").glob("*.json")):
    d = json.load(open(f))
    row = {k: d[k] for k in ("field", "bc", "obs", "window", "model", "alpha_fit", "r_fit", "vs_fit")}
    row["roi_L2_pct"] = 100 * d["roi_L2"]
    for k, v in d["chi2"].items():
        row[f"chi2n_{k}"] = v["chi2"] / v["n"]
        row[f"chi2_{k}"] = v["chi2"]
    row["active_bound"] = d["fit"]["active_bound"]
    rows.append(row)
df = pd.DataFrame(rows)
diff = df[df.model == "diff"].set_index(["field", "bc", "obs", "window"])
for i, r in df[df.model != "diff"].iterrows():
    key = (r.field, r.bc, r.obs, r.window)
    if key in diff.index:
        df.loc[i, "delta_chi2_fitobs"] = diff.loc[key, f"chi2_{r.obs}"] - r[f"chi2_{r.obs}"]
df.to_csv(twin.REPO / "results/nd/q4_summary.csv", index=False)
cols = [c for c in ["field", "bc", "obs", "window", "model", "alpha_fit", "r_fit", "vs_fit", "roi_L2_pct",
                    "chi2n_roi", "chi2n_depth", "chi2n_block8", "chi2n_field", "delta_chi2_fitobs", "active_bound"]
        if c in df]
with open(twin.REPO / "results/nd/q4_summary.md", "w") as fh:
    fh.write(df[cols].round(4).to_markdown(index=False))
print(df[cols].round(3).to_string(index=False))
