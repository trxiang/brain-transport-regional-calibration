"""Step A at the second refinement level (about 5.9M nodes), within a 36 h job limit.

A full nonlinear fit costs ~6-8 sensitivity solves on this mesh. Instead:
  1. tensor T2 truth run;
  2. model run with sensitivities at p0 (coarser-mesh fitted coordinates), metrics at p0;
  3. one bounded Gauss-Newton step on the tissue-curve objective -> p1, model run, metrics at p1.
The fitted coordinates move by about 0.1% between meshes and variants, so one exact Gauss-Newton step
from p0 lands at the optimum to well below that. Results are written after every stage.
Usage: python stepA_level2.py <standard|mmatrix> <s_D0> <r0>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
variant, sD0, r0 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
LEVELS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
T0 = time.time()
log = lambda msg: print(f"[{(time.time() - T0) / 60:7.1f} min] {msg}", flush=True)  # noqa: E731
res = {"level": LEVELS, "variant": variant, "p0": [sD0, r0]}
outfile = OUT / f"stepA_L{LEVELS}_{variant}{'_gn' if LEVELS != 2 else ''}.json"


def save():
    json.dump(res, open(outfile, "w"), indent=2, default=float)


mesh = twin.load_base_mesh()
F = twin.base_fields()
bc = twin.Boundary(np.load(OUT / "boundary_params_recovered.npy"))
D_lit, D_ten = F["LiteratureIsotropic"], F["DataTensor"]
for _ in range(LEVELS):
    mesh, parent, interp = twin.refine_uniform(mesh)
    D_lit, D_ten = D_lit[parent], D_ten[parent]
    bc = twin.InterpolatedBoundary(bc, interp)
    del parent, interp
times = twin.official_clock()
pos = times > 0
res.update(nodes=mesh.n, tets=len(mesh.C))
log(f"mesh ready: {mesh.n} nodes, {len(mesh.C)} tets")

K_ten = mesh.stiffness(D_ten)
del D_ten
K_lit = mesh.stiffness(D_lit)
if variant == "mmatrix":
    K_ten, K_lit = twin.mmatrix_variant(K_ten), twin.mmatrix_variant(K_lit)
log("stiffness assembled")

roi = twin.TissueROI(mesh)
truth = twin.Solver(mesh, K_ten, bc, times, method="cg")
c_t = truth.run(1.1, 0.0135)
res["negmass_truth"] = twin.negative_mass_ratio(mesh, c_t)
res["min_c_truth"] = float(c_t.min())
z_t = truth.weighted(c_t)
del c_t, truth, K_ten
y = roi(z_t)
norm = np.linalg.norm(y[pos])
log("truth done")
save()

model = twin.Solver(mesh, K_lit, bc, times, method="cg")
blocks = twin.Blocks(mesh, 8.0, mesh.P.min(0) - 1e-6)


def score(tag, z):
    e = z[pos] - z_t[pos]
    retained, hidden = roi.projector_parts(e)
    res[tag] = {"roi_L2": twin.rel_l2(roi(z)[pos], y[pos]), "native_L2": twin.rel_l2(z[pos], z_t[pos]),
                "hidden_fraction": hidden / (retained + hidden),
                "block8_L2": twin.rel_l2(blocks(z)[pos], blocks(z_t)[pos])}
    res[tag]["contrast_native_over_roi"] = res[tag]["native_L2"] / res[tag]["roi_L2"]
    save()


c, cs, cr = model.run(sD0, r0, sens=True)
z = model.weighted(c)
score("at_p0", z)
log(f"p0 scored: {res['at_p0']}")
R = (roi(z)[pos] - y[pos]).ravel() / norm
J = np.stack([(roi(model.weighted(cs))[pos]).ravel(), (roi(model.weighted(cr))[pos]).ravel()], 1) / norm
del c, cs, cr, z
d, *_ = np.linalg.lstsq(J, -R, rcond=None)
p1 = np.clip(np.array([sD0, r0]) + d, [0.3, 0.0], [1.4, 0.036])
res["p1"] = p1.tolist()
res["predicted_roi_L2_at_p1"] = float(np.linalg.norm(R + J @ (p1 - [sD0, r0])))
save()
log(f"Gauss-Newton step: p1 = {p1}")

z1 = model.weighted(model.run(p1[0], p1[1]))
score("at_p1", z1)
res["s_D_fit"], res["r_fit"] = float(p1[0]), float(p1[1])
res.update({k: res["at_p1"][k] for k in ("roi_L2", "native_L2", "hidden_fraction", "block8_L2", "contrast_native_over_roi")})
res["masked"] = bool(res["roi_L2"] <= 0.01 and res["native_L2"] > 0.01 and res["hidden_fraction"] >= 0.8)
res["minutes"] = (time.time() - T0) / 60
save()
log(f"done: {json.dumps(res['at_p1'])}")
