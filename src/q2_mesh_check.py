"""Discretisation check for the flow truths (cell Peclet > 1 on the working mesh at 2-4 um/min).

The continuous problem is held fixed: diffusion tensor and velocity piecewise constant on the coarse
cells (children inherit them), CSF input the P1 interpolant of the coarse input. Only the solution
space changes, so differences are discretisation error.

  fine <truth>          truth on the uniformly refined mesh (8x cells), restricted to the coarse nodes
                        -> results/nd/truths_fine/<truth>.npy; also the upwind-stabilised coarse truth
                        -> results/nd/truths_upwind/<truth>.npy
  fit <truth> <obs>     coarse diffusion-only analyst (exact inputs) fitted to the fine truth; with the
                        coarse result of Q1/Q2 this shows whether discretisation changes the apparent alpha
  summary               flow-signal errors and fitted alphas -> results/nd/mesh_check.json/.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import q1_error_budget as q1  # noqa: E402
import twin  # noqa: E402

ND = twin.REPO / "results/nd"
TRUTHS = ("base", "flow_pial_inflow_drainage_1", "flow_pial_inflow_drainage_4", "flow_pial_to_ventricle_flux_4",
          "flow_basal_to_dorsal_4")


def fine_truth(key):
    S = nd.Setup()
    clock = twin.timer()
    spec = q1.TRUTHS[key]
    D = q1.field_of(S, spec["field"])
    fine, parent, interp = twin.refine_uniform(S.mesh)
    print("fine mesh", fine.n, "nodes", len(fine.C), "cells", flush=True)
    bc = twin.InterpolatedBoundary(twin.Boundary(S.params), interp)
    K = fine.stiffness(D[parent])
    if spec.get("flow"):
        v = np.load(ND / f"velocity_{spec['flow']}.npy")
        L, q_in, _ = nd.advection_operator(fine, v[parent])
        solver = nd.GenSolver(fine, K, bc, S.t, L=L, q_in=q_in, iterative=True)
        c, _ = solver.run(spec["sD"], spec["r"], spec["vs"])
        Lu, qu, _ = nd.advection_operator(S.mesh, v, stabilise=True)
        cu, _ = nd.GenSolver(S.mesh, S.mesh.stiffness(D), twin.Boundary(S.params), S.t, L=Lu, q_in=qu).run(
            spec["sD"], spec["r"], spec["vs"])
        (ND / "truths_upwind").mkdir(parents=True, exist_ok=True)
        np.save(ND / "truths_upwind" / f"{key}.npy", cu.astype(np.float32))
    else:
        c, _ = nd.GenSolver(fine, K, bc, S.t).run(spec["sD"], spec["r"])
    (ND / "truths_fine").mkdir(parents=True, exist_ok=True)
    np.save(ND / "truths_fine" / f"{key}.npy", c[:, : S.mesh.n].astype(np.float32))
    print(key, "seconds", round(clock(), 1))


def fit_fine(key, oname):
    S = nd.Setup()
    c_obs = np.load(ND / "truths_fine" / f"{key}.npy").astype(float)
    solver = nd.GenSolver(S.mesh, S.mesh.stiffness(q1.field_of(S, q1.TRUTHS[key]["field"])),
                          twin.Boundary(S.params), S.t)
    obs = S.roi if oname == "roi" else nd.field_obs
    tmask = S.tmask()
    y = obs(solver.weighted(c_obs))
    res = nd.fit(solver, obs, y, tmask, names=("sD", "r"), starts=((1.0, 0.02), (3.0, 0.15)))
    res.pop("c")
    out = {"truth": key, "obs": oname, "alpha_fit": res["sD"], "r_fit": res["r"], "fit": res}
    (ND / "mesh_check").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ND / "mesh_check" / f"{key}__{oname}.json", "w"), indent=2, default=float)
    print(out)


def summary():
    S = nd.Setup()
    D = nd.depth_tissue_rows(S)
    i = S.scan_idx
    w = np.sqrt(S.mesh.w / S.mesh.V)[None, :]
    load = lambda d, k: np.load(ND / d / f"{k}.npy").astype(float)[i]      # noqa: E731

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    rows, out = [], {}
    base = {"coarse": load("truths", "base"), "fine": load("truths_fine", "base")}
    for key in TRUTHS:
        r = {"truth": key}
        cc, cf = load("truths", key), load("truths_fine", key)
        r["field_rel_err"] = rel(cc * w, cf * w)
        r["roi_rel_err"] = rel(nd.roi_means(S, cc), nd.roi_means(S, cf))
        r["depth_rel_err"] = rel((D @ cc.T).T, (D @ cf.T).T)
        if key != "base":
            sc, sf = cc - base["coarse"], cf - base["fine"]              # the flow signal
            r["flow_signal_field_err"] = rel(sc * w, sf * w)
            r["flow_signal_roi_err"] = rel(nd.roi_means(S, sc), nd.roi_means(S, sf))
            if (ND / "truths_upwind" / f"{key}.npy").exists():
                su = load("truths_upwind", key) - base["coarse"]
                r["upwind_flow_signal_field_err"] = rel(su * w, sf * w)
        for o in ("roi", "field"):
            f = ND / "mesh_check" / f"{key}__{o}.json"
            if f.exists():
                r[f"alpha_fine_{o}"] = json.load(open(f))["alpha_fit"]
            qc = ND / "q1" / (f"E_{key}{'_field' if o == 'field' else ''}.json" if key != "base" else
                              f"{'A0_perfect' if o == 'roi' else 'A7_field_perfect'}.json")
            if qc.exists():
                r[f"alpha_coarse_{o}"] = json.load(open(qc))["alpha_fit"]
        rows.append(r)
        out[key] = r
    json.dump(out, open(ND / "mesh_check.json", "w"), indent=2)
    import pandas as pd
    df = pd.DataFrame(rows)
    open(ND / "mesh_check.md", "w").write(df.round(4).to_markdown(index=False))
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fine":
        fine_truth(sys.argv[2])
    elif cmd == "fit":
        fit_fine(sys.argv[2], sys.argv[3])
    elif cmd == "summary":
        summary()
