"""Fingerprint attribution: which known error leaves a misfit pattern like the one in the real data?

For each Q1 case (one error source, or flow in the truth) the misfit pattern is the observed minus the
fitted tissue mean, per tissue and depth bin, at the four sessions: a depth x time signature. The real
data give the same pattern for a Q4 fit with the same analyst choices (tensor field, curve or linear
CSF input, tissue curves, all sessions). Patterns are compared by their cosine similarity, and the
real pattern is decomposed by non-negative least squares on the synthetic signatures.

Usage: python fingerprint.py            writes results/nd/fingerprint.json and fingerprint.md
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import q1_error_budget as q1  # noqa: E402
import twin  # noqa: E402

OUT = twin.REPO / "results/nd"
SIGNATURE_CASES = ("A1_linear_bc", "A2_amp0.85", "A2_amp1.15", "A3_pv0.5", "A3_pv0.5_ell2", "A6_fix_r0",
                   "B1_literature_field", "E_flow_pial_inflow_drainage_2", "E_flow_pial_inflow_drainage_4",
                   "E_flow_pial_to_ventricle_flux_2", "E_flow_pial_to_ventricle_flux_4",
                   "E_flow_basal_to_dorsal_2", "E_flow_basal_to_dorsal_4", "C1_combo")
REAL_JOBS = ("tensor__curve__roi__all__diff", "tensor__linear__roi__all__diff")


def depth_means(S, D, c):
    """(nses, nrow) tissue-depth means of a trajectory sampled at the sessions."""
    return (D @ c.T).T / D.sum(1)[None, :]


def case_pattern(S, D, name):
    """Observed minus fitted depth means for a Q1 case, rebuilt from its stored fit (one forward solve)."""
    d = json.load(open(OUT / "q1" / f"{name}.json"))
    a = q1.CASES[name]["analyst"]
    c_true = q1.truth_trajectory(S, q1.CASES[name]["truth"])
    bc_true = twin.Boundary(S.params)
    c_obs = c_true
    if a["pv"] is not None:
        Bt = np.stack([bc_true(t) for t in S.t])
        c_obs = nd.partial_volume(S, c_obs, Bt, f0=a["pv"][0], ell=a["pv"][1])
    bc_model = bc_true if a["bc"] == "curve" else nd.sampled_linear(bc_true)
    if a["amp"] != 1.0:
        bc_model = nd.ScaledBoundary(bc_model, a["amp"])
    solver = nd.GenSolver(S.mesh, S.mesh.stiffness(q1.field_of(S, a["field"])), bc_model, S.t)
    c_fit, _ = solver.run(d["alpha_fit"], d["r_fit"])
    i = S.scan_idx
    return depth_means(S, D, c_obs[i]) - depth_means(S, D, c_fit[i]), d


def main():
    S = nd.Setup()
    D = nd.depth_tissue_rows(S)
    G = nd.gonzo_data(S)
    labels = []
    for j, tissue in enumerate(("GM", "WM", "subcortical")):
        for a, b in zip(nd.DEPTH_EDGES[:-1], nd.DEPTH_EDGES[1:]):
            sel = (S.dist >= a) & (S.dist < b)
            if (S.mesh.tissue_weights()[j] * sel).sum() > 0.002 * S.mesh.tissue_weights()[j].sum():
                labels.append(f"{tissue} {a:g}-{b:g} mm")
    real = {}
    for job in REAL_JOBS:
        f = OUT / "q4" / f"{job}.npz"
        if f.exists():
            c_fit = np.load(f)["c_fit_sessions"].astype(float)
            real[job] = depth_means(S, D, G["c"][1:]) - depth_means(S, D, c_fit)
    sig = {}
    meta = {}
    for name in SIGNATURE_CASES:
        if (OUT / "q1" / f"{name}.json").exists():
            sig[name], d = case_pattern(S, D, name)
            meta[name] = {"alpha_fit": d["alpha_fit"], "r_fit": d["r_fit"]}
            print(name, "done", flush=True)

    def unit(x):
        x = x.ravel()
        return x / np.linalg.norm(x)

    out = {"rows": labels, "sessions_h": nd.SCAN_H.tolist(), "real": {k: v.tolist() for k, v in real.items()},
           "signatures": {k: v.tolist() for k, v in sig.items()}, "meta": meta, "cosine": {}, "nnls": {}}
    singles = [k for k in sig if k != "C1_combo"]
    lines = ["| signature | alpha_fit | " + " | ".join(f"cos vs {j.split('__')[1]}" for j in real) + " |",
             "|---|---|" + "---|" * len(real)]
    for k in sig:
        cs = [float(unit(sig[k]) @ unit(real[j])) for j in real]
        out["cosine"][k] = dict(zip(real, cs))
        lines.append(f"| {k} | {meta[k]['alpha_fit']:.2f} | " + " | ".join(f"{c:+.2f}" for c in cs) + " |")
    for j, rv in real.items():
        A = np.stack([unit(sig[k]) for k in singles], 1)
        w, resid = nnls(A, rv.ravel())
        out["nnls"][j] = {"weights": dict(zip(singles, w.tolist())),
                          "explained_fraction": float(1 - resid ** 2 / np.sum(rv ** 2))}
        lines += ["", f"NNLS decomposition of the real misfit ({j}): explained "
                  f"{100 * out['nnls'][j]['explained_fraction']:.1f}% of the squared misfit",
                  "| signature | weight |", "|---|---|"]
        lines += [f"| {k} | {wk:.4f} |" for k, wk in sorted(zip(singles, w), key=lambda t: -t[1]) if wk > 0]
    lines += ["", "Real misfit (observed - fitted depth means, mmol/L) for the curve-input fit:", "",
              "| row | " + " | ".join(f"{t:.1f} h" for t in nd.SCAN_H) + " |", "|---|" + "---|" * len(nd.SCAN_H)]
    if REAL_JOBS[0] in real:
        for lab, col in zip(labels, real[REAL_JOBS[0]].T):
            lines.append(f"| {lab} | " + " | ".join(f"{x:+.4f}" for x in col) + " |")
    json.dump(out, open(OUT / "fingerprint.json", "w"), indent=1)
    open(OUT / "fingerprint.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
