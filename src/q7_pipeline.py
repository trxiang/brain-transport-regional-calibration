"""A: what does the published analysis (Vinje et al. 2023) return, and which of its choices matter?

Published choices (Vinje et al. 2023, Methods): objective = native L2 over the brain summed over the scans
(2-48 h); alpha multiplies the diffusivity in the cerebral cortex only (alpha = 1 in white matter);
group-averaged, regionally constant isotropic diffusivity; the concentration of the boundary voxel
prescribed on the exterior (Dirichlet), linearly interpolated between scans; alpha in [1, 10],
r in [1e-7, 1e-3] 1/s. Starting from our native-field analyst with the Q1 combined errors, the choices
are switched on one at a time:

  S0  exact inputs (control)                 tensor, curve input, Robin, all scans, field
  S1  combined errors (= Q1 C1_combo_field)  linear input 15% low, partial volume 0.5/1 mm, noise
  S2  + scans <= 48 h
  S3  + regional isotropic diffusivity
  S4  + alpha in the cortex only
  S5  + Dirichlet from the observed boundary-node values (replaces the CSF input and the exchange)
  S6  + published bounds                     = the published pipeline
  P_exact  S6 with exact observations (no partial volume, no noise): the pipeline structure alone
  P_nopv   S6 with noise but no partial volume
  P_roi    S6 fitted to tissue curves instead of the field

Synthetic jobs: syn__<truth>__<step> (truth base: alpha 1, r 0.018; vinje: alpha 3.5, r 0.186, alpha in
all tissue). Real jobs: real__<bc>__<step> on the Gonzo maps (S1 = our field fit with that input).

Third round (list3):
  cx2, cx3.5       truths with alpha 2 or 3.5 in the cortex only (tensor, r 0.018): how the pipeline maps alpha
  perf_Q*_l*       the perfusion truths of q8: what the pipeline reports when the excess is CSF perfusion
  D_perf_l<ell>    S6 with the perfusion term instead of alpha (alpha 1; Q, r fitted), CSF concentration of
  D_both_l<ell>    the perfusion source from the analyst's input; D_both fits alpha, r, Q together
  se__<bc>__<step> cluster-robust standard errors at the real-data fit of that step
  S6w, P_exact_w   S6 and P_exact with the volume-weighted L2 objective of the publication (the other steps
                   weight every node equally, i.e. whitened by independent node noise)

Usage: python q7_pipeline.py <job> | list | list3 | --summary      writes results/nd/q7/<job>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ext_solver as ex  # noqa: E402
import nd_common as nd  # noqa: E402
import q1_error_budget as q1  # noqa: E402
import q4_real_data as q4  # noqa: E402
import q8_perfusion as q8  # noqa: E402
import twin  # noqa: E402

OUT = nd.ND_OUT / "q7"
STEPS = {
    "S0": dict(errors=False, window=None, field="tensor", cortex=False, dirichlet=False, vinje_bounds=False),
    "S1": dict(errors=True, window=None, field="tensor", cortex=False, dirichlet=False, vinje_bounds=False),
    "S2": dict(errors=True, window=48.0, field="tensor", cortex=False, dirichlet=False, vinje_bounds=False),
    "S3": dict(errors=True, window=48.0, field="regiso", cortex=False, dirichlet=False, vinje_bounds=False),
    "S4": dict(errors=True, window=48.0, field="regiso", cortex=True, dirichlet=False, vinje_bounds=False),
    "S5": dict(errors=True, window=48.0, field="regiso", cortex=True, dirichlet=True, vinje_bounds=False),
    "S6": dict(errors=True, window=48.0, field="regiso", cortex=True, dirichlet=True, vinje_bounds=True),
    "P_exact": dict(errors=False, window=48.0, field="regiso", cortex=True, dirichlet=True, vinje_bounds=True),
    "P_nopv": dict(errors="noise", window=48.0, field="regiso", cortex=True, dirichlet=True, vinje_bounds=True),
    "P_roi": dict(errors=True, window=48.0, field="regiso", cortex=True, dirichlet=True, vinje_bounds=True, obs="roi"),
}
STEPS["S6w"] = dict(STEPS["S6"], obs="field_w")             # volume-weighted L2(Omega) objective, as published
STEPS["P_exact_w"] = dict(STEPS["P_exact"], obs="field_w")
for _l in (1, 3):
    for _m in ("perf", "both"):
        STEPS[f"D_{_m}_l{_l}"] = dict(STEPS["S6"], perf=_m, ell_p=float(_l))
REAL_STEPS = ("S1", "S2", "S3", "S4", "S5", "S6", "P_roi")
CX_TRUTHS = {"cx2": 2.0, "cx3.5": 3.5}


def jobs():
    out = [f"syn__{t}__{s}" for t in ("base", "vinje") for s in STEPS]
    out += [f"real__{bc}__{s}" for bc in ("curve", "linear") for s in REAL_STEPS
            if not (bc == "curve" and STEPS[s]["dirichlet"])]          # Dirichlet steps do not use the CSF input
    return out


def jobs3():
    out = [f"syn__{t}__{s}" for t in CX_TRUTHS for s in ("S6", "P_exact", "P_nopv")]
    out += [f"syn__{t}__{s}" for t in q8.PERF_TRUTHS for s in ("S6", "P_exact")]
    out += [f"syn__{t}__D_{m}_l{l}" for t, l in (("base", 1), ("perf_Q15_l1", 1), ("perf_Q45_l1", 1), ("perf_Q45_l3", 3))
            for m in ("perf", "both")]
    out += [f"real__linear__D_{m}_l{l}" for m in ("perf", "both") for l in (1, 3)]
    out += ["se__linear__S6"]
    return out


def jobs3b():
    return ([f"syn__{t}__S6w" for t in ("base", "vinje", "cx2", "cx3.5")] + ["real__linear__S6w"] +
            [f"syn__{t}__P_exact_w" for t in ("base", "cx3.5")])


def diffusivity(S, name):
    return S.fields["DataTensor"] if name == "tensor" else ex.regional_isotropic(S)


def syn_truth(S, a):
    """Noiseless truth trajectory and its spec."""
    if a in q1.TRUTHS:
        return q1.truth_trajectory(S, a), q1.TRUTHS[a]
    if a in q8.PERF_TRUTHS:
        Q, ell = q8.PERF_TRUTHS[a]
        return q8.truth(S, a), {"field": "tensor", "sD": 1.0, "r": 0.018, "Q": Q, "ell_p": ell}
    spec = {"field": "tensor", "sD": CX_TRUTHS[a], "r": 0.018, "alpha_in": "cortex"}
    f = q1.TRUTH_CACHE / f"{a}.npy"
    if f.exists():
        return np.load(f).astype(float), spec
    K0, K1 = ex.split_stiffness(S, S.fields["DataTensor"], 1)
    c, _ = ex.ExtSolver(S.mesh, K1, twin.Boundary(S.params), S.t, K0=K0).run(spec["sD"], spec["r"])
    q1.TRUTH_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, c.astype(np.float32))
    return c, spec


def build(kind, a, step):
    """Observations, solver and model of one (data, step) combination."""
    st = STEPS[step]
    S = nd.Setup()
    bc_true = twin.Boundary(S.params)
    if kind == "syn":
        c_true, truth = syn_truth(S, a)
        Bt = np.stack([bc_true(t) for t in S.t])
        c_obs = c_true
        if st["errors"] is True:
            c_obs = nd.partial_volume(S, c_obs, Bt, f0=0.5, ell=1.0)
        if st["errors"]:
            c_obs = nd.add_noise(c_obs, q1.NOISE_SIGMA, seed=20261101)
        bc_model = nd.ScaledBoundary(nd.sampled_linear(bc_true), 0.85) if st["errors"] is True else bc_true
    else:
        G = nd.gonzo_data(S)
        c_obs = q4.real_on_clock(S, G)
        bc_model = bc_true if a == "curve" else nd.LinearBoundary(G["times_h"], G["B_raw"])
        truth = None
    K0, K1 = ex.split_stiffness(S, diffusivity(S, st["field"]), 1 if st["cortex"] else None)
    dirichlet = ex.dirichlet_from_obs(S, c_obs, st["window"]) if st["dirichlet"] else None
    perf = ex.perfusion_weights(S, st["ell_p"]) if st.get("perf") else None
    keep_bc = not st["dirichlet"] or perf is not None                 # the perfusion source needs the CSF input
    solver = ex.ExtSolver(S.mesh, K1, bc_model if keep_bc else None, S.t, K0=K0, perf=perf, dirichlet=dirichlet)
    names = {None: ("sD", "r"), "perf": ("Q", "r"), "both": ("sD", "r", "Q")}[st.get("perf")]
    fixed = {"sD": 1.0} if st.get("perf") == "perf" else None
    return dict(S=S, st=st, c_obs=c_obs, truth=truth, solver=solver, model=ex.ExtModel(S, solver), names=names,
                fixed=fixed, ops=nd.whitened_operators(S), oname=st.get("obs", "field"), tmask=S.tmask(max_h=st["window"]),
                bounds=ex.VINJE_BOUNDS if st["vinje_bounds"] else None)


def run_se(job):
    """Cluster-robust standard errors at the fitted parameters of real__<bc>__<step>."""
    import q7_uncertainty as q7u
    _, a, step = job.split("__")
    fit = json.load(open(OUT / f"real__{a}__{step}.json"))
    b = build("real", a, step)
    p = dict(ex.DEFAULTS, **fit["fit"]["params"])
    names = [k for k in b["names"] if k not in fit["fit"]["active"]]
    out = {"job": job, "params": p, **q7u.sandwich(b["S"], b["solver"], b["model"], p, names, b["oname"],
                                                     b["tmask"], b["c_obs"])}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    print(job, out["se_iid"], {h: v["se"] for h, v in out["robust"].items()})


def run(job):
    kind, a, step = job.split("__")
    if kind == "se":
        return run_se(job)
    clock = twin.timer()
    b = build(kind, a, step)
    S, st, c_obs, solver, ops, oname, tmask = (b[k] for k in ("S", "st", "c_obs", "solver", "ops", "oname", "tmask"))
    truth = b["truth"]
    o = nd.field_obs if oname == "field_w" else ops[oname]
    y = o(solver.weighted(c_obs))
    starts = [{"sD": 1.0, "r": 0.02, "Q": 20.0}, {"sD": 3.0, "r": 0.15, "Q": 80.0}]
    res = ex.fit_ext(b["model"], solver, o, y, tmask, b["names"], starts, fixed=b["fixed"], bounds=b["bounds"])
    m = res.pop("m")
    out = {"job": job, "kind": kind, "case": a, "step": step, "spec": st, "alpha_fit": res["params"]["sD"],
           "Q_fit": res["params"]["Q"],
           "r_fit": res["params"]["r"], "truth": truth, "fit": res, "chi2": ex.chi2_all(S, solver, m, c_obs, tmask, ops),
           "roi_means_data": nd.roi_means(S, c_obs)[S.scan_idx].tolist(),
           "roi_means_fit": nd.roi_means(S, m)[S.scan_idx].tolist(), "seconds": round(clock(), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    print(job, round(out["alpha_fit"], 4), round(out["r_fit"], 4), res["active"],
          {k: round(v["chi2"] / v["n"], 3) for k, v in out["chi2"].items()}, out["seconds"])


def summary():
    import pandas as pd
    rows = []
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f))
        if d["job"].startswith("se__"):
            continue
        rows.append({"kind": d["kind"], "case": d["case"], "step": d["step"], "alpha": d["alpha_fit"], "r": d["r_fit"],
                     "Q": d.get("Q_fit", 0.0), "active": ",".join(d["fit"]["active"]),
                     **{f"chi2n_{k}": v["chi2"] / v["n"] for k, v in d["chi2"].items()}})
    df = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(STEPS)}
    df = df.sort_values(["kind", "case", "step"], key=lambda c: c.map(order) if c.name == "step" else c)
    df.to_csv(nd.ND_OUT / "q7_summary.csv", index=False)
    open(nd.ND_OUT / "q7_summary.md", "w").write(df.round(4).to_markdown(index=False))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(jobs()))
    elif sys.argv[1] == "list3":
        print("\n".join(jobs3()))
    elif sys.argv[1] == "list3b":
        print("\n".join(jobs3b()))
    elif sys.argv[1] == "--summary":
        summary()
    else:
        run(sys.argv[1])
