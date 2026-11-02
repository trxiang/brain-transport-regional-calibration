"""E: do analysis errors cancel in group comparisons, or create group differences?

Published group comparisons (sleep vs sleep deprivation: Vinje et al. 2023; Brennan et al. 2025, clearance
about one third lower after deprivation) compare fitted transport parameters between groups whose CSF tracer
kinetics may also differ (Eide et al. 2021, 2026). Truths (tensor field, alpha 1, curve-shaped CSF input
B = a tau exp(-b tau) per surface node):

  <lvl>_ref        r_ref, reference CSF kinetics                       lvl a: r_ref 0.018 1/h; b: 0.12 1/h
  <lvl>_csfslow    CSF clearance slower (b x 0.75, a x 0.75: same peak, later and longer)
  <lvl>_csffast    CSF clearance faster (b x 1.33, a x 1.33)               (level a only)
  <lvl>_rlow       clearance one third lower (r x 2/3), reference CSF kinetics
  <lvl>_rlow_csfslow  both

Analysts: exact_roi, exact_field (true input, Robin); combo_roi, combo_field (Q1 combined errors on the
group's own input); vinje (the published pipeline, q7 step S6). The question is whether the fitted
differences between a group and its reference equal the true ones.

Usage: python q9_groups.py <job> | list | truths <truth> | --summary     job = <truth>__<analyst>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ext_solver as ex  # noqa: E402
import nd_common as nd  # noqa: E402
import q1_error_budget as q1  # noqa: E402
import twin  # noqa: E402

OUT = nd.ND_OUT / "q9"
R_REF = {"a": 0.018, "b": 0.12}
CSF = {"ref": 1.0, "csfslow": 0.75, "csffast": 1.33}
ANALYSTS = ("exact_roi", "exact_field", "combo_roi", "combo_field", "vinje")


def truths():
    out = {}
    for lvl, r in R_REF.items():
        for name, fb, rr in (("ref", 1.0, r), ("csfslow", 0.75, r), ("rlow", 1.0, r * 2 / 3),
                             ("rlow_csfslow", 0.75, r * 2 / 3)) + ((("csffast", 1.33, r),) if lvl == "a" else ()):
            out[f"{lvl}_{name}"] = {"csf_factor": fb, "r": rr}
    return out


TRUTHS = truths()


def jobs():
    return [f"{t}__{a}" for t in TRUTHS for a in ANALYSTS]


def boundary(S, fb):
    p = S.params.copy()
    p[:, 0] *= fb
    p[:, 1] *= fb
    return twin.Boundary(p)


def truth(S, key):
    if key == "a_ref":
        return q1.truth_trajectory(S, "base")
    f = q1.TRUTH_CACHE / f"grp_{key}.npy"
    if f.exists():
        return np.load(f).astype(float)
    spec = TRUTHS[key]
    sol = ex.ExtSolver(S.mesh, S.mesh.stiffness(S.fields["DataTensor"]), boundary(S, spec["csf_factor"]), S.t)
    c, _ = sol.run(1.0, spec["r"])
    q1.TRUTH_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, c.astype(np.float32))
    return c


def run(job):
    tkey, analyst = job.split("__")
    spec = TRUTHS[tkey]
    S = nd.Setup()
    clock = twin.timer()
    c_true = truth(S, tkey)
    bc_true = boundary(S, spec["csf_factor"])
    ops = nd.whitened_operators(S)
    errors = analyst != "exact_roi" and analyst != "exact_field"
    c_obs, bc_model = c_true, bc_true
    if errors:
        Bt = np.stack([bc_true(t) for t in S.t])
        c_obs = nd.add_noise(nd.partial_volume(S, c_true, Bt, f0=0.5, ell=1.0), q1.NOISE_SIGMA, seed=20261101)
        bc_model = nd.ScaledBoundary(nd.sampled_linear(bc_true), 0.85)
    if analyst == "vinje":
        window, bounds = 48.0, ex.VINJE_BOUNDS
        K0, K1 = ex.split_stiffness(S, ex.regional_isotropic(S), 1)
        solver = ex.ExtSolver(S.mesh, K1, None, S.t, K0=K0, dirichlet=ex.dirichlet_from_obs(S, c_obs, window))
        oname = "field"
    else:
        window, bounds = None, None
        solver = ex.ExtSolver(S.mesh, S.mesh.stiffness(S.fields["DataTensor"]), bc_model, S.t)
        oname = analyst.split("_")[1]
    tmask = S.tmask(max_h=window)
    y = ops[oname](solver.weighted(c_obs))
    res = ex.fit_ext(ex.ExtModel(S, solver), solver, ops[oname], y, tmask, ("sD", "r"),
                     [{"sD": 1.0, "r": 0.02}, {"sD": 3.0, "r": 0.15}], bounds=bounds)
    m = res.pop("m")
    out = {"job": job, "truth": tkey, "truth_spec": spec, "analyst": analyst, "alpha_fit": res["params"]["sD"],
           "r_fit": res["params"]["r"], "fit": res, "chi2": ex.chi2_all(S, solver, m, c_obs, tmask, ops),
           "seconds": round(clock(), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    print(job, round(out["alpha_fit"], 4), round(out["r_fit"], 4), res["active"], out["seconds"])


def summary():
    import pandas as pd
    rows = [json.load(open(f)) for f in sorted(OUT.glob("*.json"))]
    df = pd.DataFrame([{"truth": d["truth"], "analyst": d["analyst"], "r_true": d["truth_spec"]["r"],
                        "csf_factor": d["truth_spec"]["csf_factor"], "alpha": d["alpha_fit"], "r": d["r_fit"]}
                       for d in rows])
    out = []
    for _, g in df.iterrows():
        lvl = g.truth.split("_")[0]
        ref = df[(df.truth == f"{lvl}_ref") & (df.analyst == g.analyst)]
        if not len(ref) or g.truth.endswith("_ref"):
            continue
        ref = ref.iloc[0]
        out.append({"truth": g.truth, "analyst": g.analyst, "alpha_ref": ref.alpha, "alpha_grp": g.alpha,
                    "dalpha_pct": 100 * (g.alpha / ref.alpha - 1), "r_ref": ref.r, "r_grp": g.r,
                    "dr_pct_fit": 100 * (g.r / ref.r - 1) if ref.r > 0 else np.nan,
                    "dr_pct_true": 100 * (g.r_true / TRUTHS[f"{lvl}_ref"]["r"] - 1)})
    dd = pd.DataFrame(out)
    df.to_csv(nd.ND_OUT / "q9_fits.csv", index=False)
    dd.to_csv(nd.ND_OUT / "q9_summary.csv", index=False)
    open(nd.ND_OUT / "q9_summary.md", "w").write(df.round(4).to_markdown(index=False) + "\n\n" +
                                                 dd.round(2).to_markdown(index=False))
    print(df.round(4).to_string(index=False))
    print(dd.round(2).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(jobs()))
    elif sys.argv[1] == "truths":
        truth(nd.Setup(), sys.argv[2])
    elif sys.argv[1] == "--summary":
        summary()
    else:
        run(sys.argv[1])
