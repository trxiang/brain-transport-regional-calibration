"""Step C2 analysis: fitted s_D versus correlation length, against predictions from the true field."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import homog  # noqa: E402
import twin  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
mesh = twin.load_base_mesh()
F = twin.base_fields()
D_lit = F["LiteratureIsotropic"]
T2 = np.load(twin.EXT / "04_DATA/trajectories/T2.npz")
c = T2["NATIVE"].astype(float) / np.sqrt(mesh.w / mesh.V)
Ew = homog.gradient_energy_weights(mesh, c, T2["times_hours"], D_lit)
cache = {}
rows = []
fields = {f"ell{e:g}": D_lit * twin.correlated_multiplier(mesh, e) for e in (1, 2, 4, 8, 16, 32)}
fields["E3"] = D_lit * twin.atlas_multiplier("E3", mesh)
fields["E4"] = D_lit * twin.atlas_multiplier("E4", mesh)
fields["orient"] = twin.orientation_only_tensor(D_lit, F["DataTensor"])
for key, D in fields.items():
    row = {"truth": key, **homog.predictions(mesh, D, D_lit, Ew, lit_cache=cache)}
    for lev in ("", "_L1"):
        f = RES / "cases" / f"C2_{key}{lev}.json"
        if f.exists():
            r = json.load(open(f))
            for k in ("s_D_fit", "r_fit", "roi_L2", "native_L2", "hidden_fraction"):
                row[f"{k}{lev}"] = r[k]
    rows.append(row)
    print(row, flush=True)
pd.DataFrame(rows).to_csv(RES / "stepC2_scale.csv", index=False)
