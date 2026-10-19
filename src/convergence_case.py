"""Mesh convergence of the headline regimes beyond T2 (tensor T1, T3 and the data-isotropic E5).

Cases (truth field, truth coordinates, s_D lower bound of the canonical fit):
  T1      tensor,        (1.3, 0.0045), 0.6
  T2      tensor,        (1.1, 0.0135), 0.6
  T3      tensor,        (0.7, 0.0315), 0.6   (the paper's bound; the fit touches it)
  T3wide  tensor,        (0.7, 0.0315), 0.3   (Step D1 bound)
  E5      DataIsotropic, (1.1, 0.0135), 0.6
Modes:
  fit  full bounded nonlinear fit (levels 0 and 1); start from x0 or the next-coarser result.
  gn   level 2: truth, model with sensitivities at p0 (the level-1 fit), one bounded linearised
       least-squares step (lsq_linear) to p1, model at p1. Results are written after every stage.
Usage: python convergence_case.py <case> <level> <fit|gn>
Writes results/convergence/<case>_L<level>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402

CASES = {"T1": ("tensor", 1.3, 0.0045, 0.6), "T2": ("tensor", 1.1, 0.0135, 0.6),
         "T3": ("tensor", 0.7, 0.0315, 0.6), "T3wide": ("tensor", 0.7, 0.0315, 0.3),
         "E5": ("dataiso", 1.1, 0.0135, 0.6)}
UB = (1.4, 0.036)
ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "convergence"
OUT.mkdir(exist_ok=True)
case, level, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
field, sD_true, r_true, lb_s = CASES[case]
LB = (lb_s, 0.0)
T0 = time.time()
log = lambda msg: print(f"[{(time.time() - T0) / 60:7.1f} min] {msg}", flush=True)  # noqa: E731
res = {"case": case, "level": level, "mode": mode, "truth_field": field, "s_D_true": sD_true, "r_true": r_true,
       "lb": LB, "ub": UB}
outfile = OUT / f"{case}_L{level}.json"


def save():
    json.dump(res, open(outfile, "w"), indent=2, default=float)


def start_point():
    """Fitted coordinates of the next-coarser level (or the archived level-0 values)."""
    prev = OUT / f"{case}_L{level - 1}.json"
    if level > 0 and prev.exists():
        d = json.load(open(prev))
        return float(d["s_D_fit"]), float(d["r_fit"])
    return {"T1": (1.0466, 0.00577), "T2": (0.8911, 0.01496), "T3": (0.6, 0.0353),
            "T3wide": (0.574, 0.0335), "E5": (0.8779, 0.0143)}[case]


mesh = twin.load_base_mesh()
F = twin.base_fields()
bc = twin.Boundary(np.load(ROOT / "boundary_params_recovered.npy"))
D_lit, D_true = F["LiteratureIsotropic"], F["DataTensor" if field == "tensor" else "DataIsotropic"]
for _ in range(level):
    mesh, parent, interp = twin.refine_uniform(mesh)
    D_lit, D_true = D_lit[parent], D_true[parent]
    bc = twin.InterpolatedBoundary(bc, interp)
    del parent, interp
times = twin.official_clock()
pos = times > 0
res.update(nodes=mesh.n, tets=len(mesh.C))
log(f"{case} level {level}: {mesh.n} nodes")

K_true = mesh.stiffness(D_true)
del D_true
K_lit = mesh.stiffness(D_lit)
roi = twin.TissueROI(mesh)
truth = twin.Solver(mesh, K_true, bc, times, method="cg")
c_t = truth.run(sD_true, r_true)
res["negmass_truth"] = twin.negative_mass_ratio(mesh, c_t)
res["min_c_truth"] = float(c_t.min())
z_t = truth.weighted(c_t)
del c_t, truth, K_true
y = roi(z_t)
norm = np.linalg.norm(y[pos])
log("truth done")
save()

model = twin.Solver(mesh, K_lit, bc, times, method="cg")
blocks = twin.Blocks(mesh, 8.0, mesh.P.min(0) - 1e-6)


def score(z):
    e = z[pos] - z_t[pos]
    retained, hidden = roi.projector_parts(e)
    m = {"roi_L2": twin.rel_l2(roi(z)[pos], y[pos]), "native_L2": twin.rel_l2(z[pos], z_t[pos]),
         "hidden_fraction": hidden / (retained + hidden),
         "block8_L2": twin.rel_l2(blocks(z)[pos], blocks(z_t)[pos])}
    m["contrast_native_over_roi"] = m["native_L2"] / m["roi_L2"]
    return m


p0 = start_point()
res["p0"] = list(p0)
if mode == "fit":
    fit = twin.fit_two_scalar(model, roi, y, x0=p0, lb=LB, ub=UB)
    res.update(s_D_fit=fit["s_D"], r_fit=fit["r"], nfev=fit["nfev"], active_bound=fit["active_bound"])
    res.update(score(model.weighted(fit["c"])))
    res["negmass_fit"] = twin.negative_mass_ratio(mesh, fit["c"])
elif mode == "gn":
    c, cs, cr = model.run(*p0, sens=True)
    z = model.weighted(c)
    res["at_p0"] = score(z)
    save()
    log(f"p0 scored: {res['at_p0']}")
    R = (roi(z)[pos] - y[pos]).ravel() / norm
    J = np.stack([roi(model.weighted(cs))[pos].ravel(), roi(model.weighted(cr))[pos].ravel()], 1) / norm
    del c, cs, cr, z
    scale = np.array([UB[0] - LB[0], UB[1] - LB[1]])           # same scaling as fit_two_scalar
    step = lsq_linear(J * scale, -R, bounds=((np.array(LB) - p0) / scale, (np.array(UB) - p0) / scale))
    p1 = np.array(p0) + step.x * scale
    res["p1"] = p1.tolist()
    res["active_bound"] = bool(np.any(step.active_mask != 0))
    res["predicted_roi_L2_at_p1"] = float(np.linalg.norm(R + J @ (p1 - p0)))
    save()
    log(f"bounded Gauss-Newton step: p1 = {p1}")
    res["at_p1"] = score(model.weighted(model.run(*p1)))
    res.update(s_D_fit=float(p1[0]), r_fit=float(p1[1]), **res["at_p1"])
else:
    raise SystemExit(f"unknown mode {mode}")
res["masked"] = bool(res["roi_L2"] <= 0.01 and res["native_L2"] > 0.01 and res["hidden_fraction"] >= 0.8)
res["minutes"] = (time.time() - T0) / 60
save()
log(f"done: s_D {res['s_D_fit']:.4f} r {res['r_fit']:.5f} roi {res['roi_L2']:.4%} native {res['native_L2']:.4%} "
    f"hidden {res['hidden_fraction']:.2%}")
