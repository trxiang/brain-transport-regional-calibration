"""Step 0: validate the Python twin against every archived DuMux trajectory.

Pass rule (set before running): pooled relative L2 error of the weighted native field
below 1e-3 for every trajectory."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"

mesh = twin.load_base_mesh()
fields = twin.base_fields()
Bdata = twin.checkpoint_boundary_data()
bmask = mesh.a > 0
params, recovery_misfit = twin.recover_official_boundary(mesh, fields["LiteratureIsotropic"], "L22")
np.save(OUT / "boundary_params_recovered.npy", params)
own = twin.fit_boundary_curve(Bdata, bmask)
np.save(OUT / "boundary_params_ownfit.npy", own)
bc = twin.Boundary(params)
times = twin.official_clock()

# consistency of the dual volumes and tissue weights with the archived operators
ops = np.load(twin.CORE / "04_DATA/observation_operators/operators.npz")
geom = {
    "node_volume_max_rel_diff": float(np.max(np.abs(mesh.w - ops["node_volume_mm3"]) / ops["node_volume_mm3"])),
    "tissue_weight_max_abs_diff": float(np.max(np.abs(mesh.tissue_weights() - ops["tissue_node_volume_mm3"]))),
    "boundary_nodes_twin": int(bmask.sum()),
    "boundary_nodes_data_nonzero": int((np.abs(Bdata).sum(0) > 0).sum()),
    "boundary_recovery_misfit_rel": recovery_misfit,
    "boundary_recovered_from": "L22 (excluded from the pass test)",
    "own_checkpoint_fit_vertices_differing_gt_1e-3": int((np.abs(params[bmask] - own[bmask]).max(1) > 1e-3).sum()),
}
print(geom)

man = pd.read_csv(twin.EXT / "03_RESULTS/phase8c_library_manifest.csv").set_index("id")
cases = [(i, "LiteratureIsotropic", man.loc[i, "s_D"], man.loc[i, "r_eff_h_inv"], twin.EXT)
         for i in man.index if i.startswith("L")]
cases += [(i, "LiteratureIsotropic", man.loc[i, "s_D"], man.loc[i, "r_eff_h_inv"], twin.EXT) for i in ("T1", "T2", "T3")]
cases += [("DataTensor_T1", "DataTensor", 1.3, 0.0045, twin.EXT),
          ("DataTensor_T2", "DataTensor", 1.1, 0.0135, twin.CORE),
          ("DataTensor_T3", "DataTensor", 0.7, 0.0315, twin.EXT),
          ("DataIsotropic_T2", "DataIsotropic", 1.1, 0.0135, twin.EXT)]
ex = pd.read_csv(twin.CORE / "03_RESULTS/phase8e_exact_checks.csv").set_index("id")
cases += [("Exact8E_" + reg, "LiteratureIsotropic", ex.loc[f"DataTensor_{reg}", "estimated_s_D"],
           ex.loc[f"DataTensor_{reg}", "estimated_r_h_inv"], twin.EXT) for reg in ("T1", "T2")]

Ks = {name: mesh.stiffness(D) for name, D in fields.items()}
solvers = {name: twin.Solver(mesh, K, bc, times) for name, K in Ks.items()}
roi = twin.TissueROI(mesh, ops["tissue_node_volume_mm3"])
rows = []
for cid, mode, sD, r, base in cases:
    clock = twin.timer()
    ref = np.load(base / f"04_DATA/trajectories/{cid}.npz")
    assert np.allclose(ref["times_hours"], times)
    z_ref = ref["NATIVE"].astype(float)
    z = solvers[mode].weighted(solvers[mode].run(float(sD), float(r)))
    pos = times > 0
    per_t = np.linalg.norm(z[pos] - z_ref[pos], axis=1) / np.linalg.norm(z_ref[pos], axis=1)
    rows.append({"id": cid, "mode": mode, "s_D": float(sD), "r_h": float(r),
                 "native_rel_L2": twin.rel_l2(z[pos], z_ref[pos]),
                 "native_max_time_rel_L2": float(per_t.max()),
                 "roi_rel_L2": twin.rel_l2(roi(z)[pos], ref["TISSUE_ROI"].astype(float)[pos]),
                 "min_c_twin": float((z / np.sqrt(mesh.w / mesh.V)).min()),
                 "seconds": round(clock(), 1)})
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "step0_validation.csv", index=False)
summary = {"geometry": geom, "n_trajectories": len(df),
           "max_native_rel_L2": float(df.native_rel_L2.max()),
           "max_roi_rel_L2": float(df.roi_rel_L2.max()),
           "pass_rule": "native_rel_L2 < 1e-3 for every trajectory",
           "PASS": bool((df[df.id != "L22"].native_rel_L2 < 1e-3).all())}
json.dump(summary, open(OUT / "step0_validation_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
