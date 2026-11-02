"""Q4: calibrate the model to the real Gonzo tracer concentrations.

Each job fits the enhancement factor alpha (s_D) and clearance r (and, for flow models, a flow speed)
to the real concentration maps at the four post-injection sessions, under one combination of the
analysis choices published studies make:

  field   tensor | literature | dataiso          diffusion field multiplied by alpha
  bc      curve | linear                         miniapp curve fit, or raw CSF values linearly interpolated
  obs     roi | depth | field                    fit target (whitened, sigma = 0.02 mmol/L per node)
  window  all | 24                               sessions used
  model   diff | flow-<pattern>                  diffusion only, or with a flow pattern of free speed

After the fit, the misfit is evaluated in every observation space as chi^2 against its noise
expectation (the number of observations), so a regional fit can be checked against the depth
profiles and the native field it did not use.

Usage: python q4_real_data.py <job> | list     job = <field>__<bc>__<obs>__<window>__<model>
Writes results/nd/q4/<job>.json (and the fitted trajectory at the sessions as .npz).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import q1_error_budget as q1  # noqa: E402
import twin  # noqa: E402

OUT = twin.REPO / "results/nd/q4"
FLOWS = ("pial_inflow_drainage", "pial_to_ventricle_flux", "basal_to_dorsal")


def jobs():
    out = []
    for field in ("tensor", "literature", "dataiso"):
        for bc in ("curve", "linear"):
            for obs in ("roi", "depth", "field"):
                for win in ("all", "24"):
                    out.append(f"{field}__{bc}__{obs}__{win}__diff")
    for bc in ("curve", "linear"):
        for obs in ("roi", "depth", "field"):
            for F in FLOWS:
                out.append(f"tensor__{bc}__{obs}__all__flow-{F}")
    return out


def real_on_clock(S, G):
    """Real concentrations placed on the solver clock (only the session rows are used)."""
    c = np.zeros((len(S.t), S.mesh.n))
    c[S.scan_idx] = G["c"][1:]
    return c


def run(job):
    field, bc, oname, win, model = job.split("__")
    S = nd.Setup()
    G = nd.gonzo_data(S)
    clock = twin.timer()
    c_obs = real_on_clock(S, G)
    bc_model = twin.Boundary(S.params) if bc == "curve" else nd.LinearBoundary(G["times_h"], G["B_raw"])
    K = S.mesh.stiffness(q1.field_of(S, field))
    ops = nd.whitened_operators(S)
    obs = ops[oname]
    tmask = S.tmask(max_h=24.0 if win == "24" else None)
    if model == "diff":
        solver = nd.GenSolver(S.mesh, K, bc_model, S.t)
        names, starts = ("sD", "r"), ((1.0, 0.02), (3.0, 0.15))
    else:
        v = np.load(twin.REPO / f"results/nd/velocity_{model.split('-', 1)[1]}.npy")
        L, q_in, _ = nd.advection_operator(S.mesh, v)
        solver = nd.GenSolver(S.mesh, K, bc_model, S.t, L=L, q_in=q_in)
        prior = OUT / f"{field}__{bc}__{oname}__{win}__diff.json"
        a0, r0 = (1.0, 0.02)
        if prior.exists():
            d = json.load(open(prior))
            a0, r0 = d["alpha_fit"], d["r_fit"]
        names, starts = ("sD", "r", "vs"), ((a0, r0, 1.0),)
    y = obs(solver.weighted(c_obs))
    res = nd.fit(solver, obs, y, tmask, names=names, starts=starts, max_nfev=60)
    c_fit = res.pop("c")
    chi2 = {}
    for k, o in ops.items():
        r_ = o(solver.weighted(c_fit))[tmask] - o(solver.weighted(c_obs))[tmask]
        chi2[k] = {"chi2": float(np.sum(r_ ** 2)), "n": int(o.nrow * tmask.sum())}
    w = np.sqrt(S.mesh.w / S.mesh.V)[None, :]
    out = {"job": job, "field": field, "bc": bc, "obs": oname, "window": win, "model": model,
           "alpha_fit": res["sD"], "r_fit": res["r"], "vs_fit": res["vs"], "fit": res, "chi2": chi2,
           "roi_L2": twin.rel_l2(S.roi(c_fit[tmask] * w), S.roi(c_obs[tmask] * w)),
           "native_L2": twin.rel_l2(c_fit[tmask] * w, c_obs[tmask] * w),
           "data_roi_means": nd.roi_means(S, c_obs)[S.scan_idx].tolist(),
           "fit_roi_means": nd.roi_means(S, c_fit)[S.scan_idx].tolist(),
           "seconds": round(clock(), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    np.savez_compressed(OUT / f"{job}.npz", c_fit_sessions=c_fit[S.scan_idx].astype(np.float32))
    print(json.dumps({k: out[k] for k in ("job", "alpha_fit", "r_fit", "vs_fit", "roi_L2", "seconds")}, default=float),
          {k: round(v["chi2"] / v["n"], 3) for k, v in chi2.items()})


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(jobs()))
    else:
        run(sys.argv[1])
