"""Compare the central Q1, Q2 and Q5 results across geometries (Gonzo, Vinje meshes 14 and 4).

Usage: python geometry_compare.py      writes results/nd/geometry_compare.csv and .md
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402

GEOMS = {"gonzo": twin.REPO / "results/nd", "vinje14": twin.REPO / "results/nd_vinje14",
         "vinje4": twin.REPO / "results/nd_vinje4"}
Q1 = ["A0_perfect", "A1_linear_bc", "A2_amp0.85", "A3_pv0.5", "A3_pv0.5_ell2", "A6_fix_r0", "B1_literature_field",
      "C1_combo", "C1_combo_field", "D_vinje_combo", "E_flow_pial_inflow_drainage_2",
      "E_flow_pial_inflow_drainage_2_field", "E_flow_pial_inflow_drainage_4"]
Q2 = [("base", "combo", "pial_inflow_drainage", o) for o in ("roi", "depth", "field")] + \
     [("flow_pial_inflow_drainage_1", "exact", "pial_inflow_drainage", o) for o in ("roi", "field")] + \
     [("flow_pial_to_ventricle_flux_1", "exact", "pial_to_ventricle_flux", "roi")]
Q5 = ["syn__base__roi__base", "syn__base__roi__amp_pv", "syn__base__field__amp_pv", "syn__vinje__roi__amp_pv"]


def load(p):
    return json.load(open(p)) if p.exists() else None


rows = []
for case in Q1:
    r = {"study": "Q1", "case": case, "quantity": "alpha_fit"}
    for g, d in GEOMS.items():
        j = load(d / "q1" / f"{case}.json")
        r[g] = j["alpha_fit"] if j else None
    rows.append(r)
for truth, err, pat, o in Q2:
    for q in ("vs_fit", "delta_chi2"):
        r = {"study": "Q2", "case": f"{truth}__{err}__{pat}__{o}", "quantity": q}
        for g, d in GEOMS.items():
            jd = load(d / "q2" / f"{truth}__{err}__diff__{o}.json")
            jf = load(d / "q2" / f"{truth}__{err}__flow-{pat}__{o}.json")
            if jd and jf:
                # vs multiplies a 1 um/min field, so it is the speed in um/min
                r[g] = jf["vs_fit"] if q == "vs_fit" else jd["chi2_noisefree"] - jf["chi2_noisefree"]
            else:
                r[g] = None
        rows.append(r)
for job in Q5:
    for q in ("sD", "amp", "f0"):
        r = {"study": "Q5", "case": job, "quantity": q}
        for g, d in GEOMS.items():
            j = load(d / "q5" / f"{job}.json")
            r[g] = j[q] if j else None
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(twin.REPO / "results/nd/geometry_compare.csv", index=False)
open(twin.REPO / "results/nd/geometry_compare.md", "w").write(df.round(4).to_markdown(index=False))
print(df.round(3).to_string(index=False))
