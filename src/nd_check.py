"""Checks for nd_common.py: reproduction of twin.Solver, advection consistency and conservation,
sensitivities against finite differences, velocity-field statistics. Writes results/nd/check.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import twin  # noqa: E402

OUT = twin.REPO / "results/nd"
OUT.mkdir(parents=True, exist_ok=True)
S = nd.Setup()
m = S.mesh
D = S.fields["DataTensor"]
K = m.stiffness(D)
bc = twin.Boundary(S.params)
out = {"nodes": m.n, "outer_nodes": len(S.outer_nodes), "ventricle_nodes": len(S.vent_nodes)}

# 1. vs = 0 reproduces twin.Solver
c_ref = twin.Solver(m, K, bc, S.t).run(1.1, 0.0135)
c_new, _ = nd.GenSolver(m, K, bc, S.t).run(1.1, 0.0135)
out["reproduce_twin_rel"] = twin.rel_l2(c_new[1:], c_ref[1:])

# 2. velocity fields
z = m.P[:, 2]
zo = z[S.outer_nodes]
fields = {
    "pial_to_ventricle": (S.outer_nodes, S.vent_nodes),
    "ventricle_to_pial": (S.vent_nodes, S.outer_nodes),
    "basal_to_dorsal": (S.outer_nodes[zo < np.quantile(zo, 0.2)], S.outer_nodes[zo > np.quantile(zo, 0.8)]),
}
for name, (hi, lo) in fields.items():
    p, v = nd.darcy_field(m, hi, lo)
    L, q_in, q = nd.advection_operator(m, v)
    interior = np.setdiff1d(np.arange(m.n), np.concatenate([hi, lo]))
    pe = nd.cell_peclet(m, v * 2.0, D)                         # at 2 um/min
    speed = np.linalg.norm(v, axis=1) / nd.UM_PER_MIN
    out[name] = {
        "net_flux_rel": float(abs(q.sum()) / np.abs(q).sum()),
        "interior_flux_max_rel": float(np.abs(q[interior]).max() / np.abs(q).max()),
        "rowsum_minus_qin_max": float(np.abs(L @ np.ones(m.n) - q_in).max() / q_in.max()),
        "speed_um_min_p50_p95_max": [float(np.percentile(speed, 50)), float(np.percentile(speed, 95)),
                                     float(speed.max())],
        "frac_cells_Pe_gt1_at_2um": float(np.average(pe > 1, weights=m.vol)),
        "inflow_area_frac": float(m.a[q_in > 0].sum() / m.a.sum()),
    }
    np.save(OUT / f"velocity_{name}.npy", v)

# 3. sensitivities against central differences (pial_to_ventricle field, 2 um/min)
v = np.load(OUT / "velocity_pial_to_ventricle.npy")
L, q_in, _ = nd.advection_operator(m, v)
solver = nd.GenSolver(m, K, bc, S.t, L=L, q_in=q_in)
clock = twin.timer()
c0, Sens = solver.run(1.0, 0.018, 2.0, sens=("sD", "r", "vs"))
out["seconds_forward_plus_3_sens_lu"] = round(clock(), 1)
w = S.roi
for name, base, h in (("sD", 1.0, 1e-4), ("r", 0.018, 1e-5), ("vs", 2.0, 1e-4)):
    args = {"sD": 1.0, "r": 0.018, "vs": 2.0}
    args[name] = base + h
    cp, _ = solver.run(**args)
    args[name] = base - h
    cm, _ = solver.run(**args)
    fd = (cp - cm) / (2 * h)
    out[f"sens_{name}_rel_err"] = twin.rel_l2(Sens[name][1:], fd[1:])

# 4. mass balance with advection (amounts, backward Euler new state)
ph = twin.Physics()
M = (m.w[None] * c0).sum(1)
err = []
for i in range(1, len(S.t)):
    dt = S.t[i] - S.t[i - 1]
    B = solver.Bt[i]
    robin = (m.a * ph.k * (B / ph.phi_sas - c0[i] / ph.phi_brain)).sum()
    adv_in = 2.0 * (q_in * ph.phi_brain * B / ph.phi_sas).sum()
    Lc = 2.0 * (L @ c0[i])                                      # sums to the advective outflow
    sink = 0.018 * (m.w * c0[i]).sum()
    dM = (M[i] - M[i - 1]) / dt
    err.append(abs(dM - (robin + adv_in - Lc.sum() - sink)))
out["mass_balance_max_rel"] = float(max(err) / np.abs(np.diff(M) / np.diff(S.t)).max())
out["negmass_ratio_2um"] = twin.negative_mass_ratio(m, c0)

print(json.dumps(out, indent=2, default=float))
json.dump(out, open(OUT / "check.json", "w"), indent=2, default=float)
