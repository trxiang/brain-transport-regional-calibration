"""Run one named case of Steps B, C2, C3 or D1 and write results/cases/<name>.json.

Every case: generate a truth with a chosen diffusion field (T2 coordinates s_D=1.1, r=0.0135 1/h
unless stated), fit the canonical two-scalar model to the three tissue curves, score it.
Usage: python run_case.py <case> [level]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402
from experiments import fit_and_score  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "cases"
OUT.mkdir(parents=True, exist_ok=True)
name = sys.argv[1]
level = int(sys.argv[2]) if len(sys.argv) > 2 else 0
METHOD = "cg" if level == 0 else "amg"

mesh = twin.load_base_mesh()
F = twin.base_fields()
params = np.load(Path(__file__).resolve().parents[1] / "results/boundary_params_recovered.npy")
bc = twin.Boundary(params)
D_lit, D_ten, D_iso = F["LiteratureIsotropic"], F["DataTensor"], F["DataIsotropic"]
m_E3, m_E4 = twin.atlas_multiplier("E3", mesh), twin.atlas_multiplier("E4", mesh)
base_for_fields = mesh
interp_chain = []
for _ in range(level):
    mesh, parent, interp = twin.refine_uniform(mesh)
    D_lit, D_ten, D_iso, m_E3, m_E4 = D_lit[parent], D_ten[parent], D_iso[parent], m_E3[parent], m_E4[parent]
    bc = twin.InterpolatedBoundary(bc, interp)
times = twin.official_clock()
pos = times > 0


def truth_field(key):
    if key == "tensor":
        return D_ten
    if key == "dataiso":
        return D_iso
    if key == "E2":
        return D_lit * twin.e2_multiplier(mesh)
    if key == "E3":
        return D_lit * m_E3
    if key == "E4":
        return D_lit * m_E4
    if key.startswith("ell"):
        return D_lit * twin.correlated_multiplier(mesh, float(key[3:]))
    if key == "orient":
        return twin.orientation_only_tensor(D_lit, D_ten)
    raise KeyError(key)


def simulate(K, sD, r, boundary=None, phys=twin.Physics()):
    s = twin.Solver(mesh, K, boundary or bc, times, phys=phys, method=METHOD)
    c = s.run(sD, r)
    return c, s.weighted(c)


K_lit = mesh.stiffness(D_lit)
result = {"case": name, "level": level, "nodes": mesh.n}
clock = twin.timer()

if name.startswith("B_"):                                    # Step B: convention sensitivity
    k, phib, phis = twin.K_SURF, twin.PHI_BRAIN, twin.PHI_SAS
    opt = name[2:]
    if opt == "k0.5":
        k *= 0.5
    elif opt == "k2":
        k *= 2
    elif opt == "phib0.1":
        phib = 0.1
    elif opt == "phib0.4":
        phib = 0.4
    elif opt == "phisas1":
        phis = 1.0
    elif opt != "base":
        raise KeyError(opt)
    phys = twin.Physics(k=k, phi_brain=phib, phi_sas=phis)
    c_t, z_t = simulate(mesh.stiffness(D_ten), 1.1, 0.0135, phys=phys)
    out, _, _ = fit_and_score(mesh, K_lit, bc, times, z_t, phys=phys, method=METHOD, x0=(0.9, 0.015))
    result.update(out, k_mm_h=k, phi_brain=phib, phi_sas=phis, peak_mean_c=float((c_t * mesh.w).sum(1).max() / mesh.V))

elif name.startswith("C2_") or name.startswith("D1_") or name.startswith("A_"):   # scale series / bounds
    key = name.split("_", 1)[1]
    sD_true, r_true, lb, x0 = 1.1, 0.0135, (0.6, 0.0), (1.0, 0.0135)
    if name.startswith("D1_"):                             # tensor T3 with a wider s_D range
        key, sD_true, r_true, lb, x0 = "tensor", 0.7, 0.0315, (0.3, 0.0), (0.6, 0.03)
    D = truth_field(key)
    c_t, z_t = simulate(mesh.stiffness(D), sD_true, r_true)
    out, _, _ = fit_and_score(mesh, K_lit, bc, times, z_t, method=METHOD, x0=x0, lb=lb)
    result.update(out, truth=key, s_D_true=sD_true, r_true=r_true)

elif name.startswith("C3a_"):                               # time-window transfer
    key = name[4:]
    c_t, z_t = simulate(mesh.stiffness(truth_field(key)), 1.1, 0.0135)
    roi = twin.TissueROI(mesh)
    early, late = (times > 0) & (times <= 24.0), times > 24.0
    for tag, mask in (("full", pos), ("early", early), ("late", late)):
        out, z_fit, _ = fit_and_score(mesh, K_lit, bc, times, z_t, method=METHOD, tmask=mask, x0=(1.0, 0.0135))
        result[tag] = {k: out[k] for k in ("s_D_fit", "r_fit", "roi_L2", "active_bound")}
        result[tag]["late_roi_L2"] = twin.rel_l2(roi(z_fit)[late], roi(z_t)[late])
        result[tag]["late_native_L2"] = twin.rel_l2(z_fit[late], z_t[late])
    result["truth"] = key

elif name.startswith("C3b_"):                               # inflow transfer
    key, inflow = name[4:].rsplit("_", 1)
    if inflow == "fast":
        bc_new = twin.Boundary(params, time_scale=2.0)
    elif inflow == "partial":
        P0 = twin.load_base_mesh().P
        bnd = params[:, 0] != 0
        zmed = np.median(P0[bnd, 2])
        bc_new = twin.Boundary(params, region=(P0[:, 2] < zmed).astype(float))
    else:
        raise KeyError(inflow)
    if level:
        raise NotImplementedError("C3b runs on the base mesh")
    K_t = mesh.stiffness(truth_field(key))
    roi = twin.TissueROI(mesh)
    _, z_std = simulate(K_t, 1.1, 0.0135)
    fit_std, _, _ = fit_and_score(mesh, K_lit, bc, times, z_std, method=METHOD, x0=(1.0, 0.0135))
    _, z_new = simulate(K_t, 1.1, 0.0135, boundary=bc_new)
    _, z_pred = simulate(K_lit, fit_std["s_D_fit"], fit_std["r_fit"], boundary=bc_new)
    fit_new, _, _ = fit_and_score(mesh, K_lit, bc_new, times, z_new, method=METHOD,
                                  x0=(fit_std["s_D_fit"], fit_std["r_fit"]))
    result.update(truth=key, inflow=inflow,
                  s_D_fit_standard=fit_std["s_D_fit"], r_fit_standard=fit_std["r_fit"],
                  roi_L2_standard=fit_std["roi_L2"], native_L2_standard=fit_std["native_L2"],
                  transfer_roi_L2=twin.rel_l2(roi(z_pred)[pos], roi(z_new)[pos]),
                  transfer_native_L2=twin.rel_l2(z_pred[pos], z_new[pos]),
                  s_D_refit_new=fit_new["s_D_fit"], r_refit_new=fit_new["r_fit"],
                  roi_L2_refit_new=fit_new["roi_L2"], native_L2_refit_new=fit_new["native_L2"])
else:
    raise KeyError(name)

result["seconds"] = round(clock(), 1)
print(json.dumps(result, indent=2, default=float))
json.dump(result, open(OUT / f"{name}{'' if level == 0 else f'_L{level}'}.json", "w"), indent=2, default=float)
