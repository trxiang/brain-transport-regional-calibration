"""Q2: can tracer-MRI observations tell tissue flow from enhanced diffusion?

For each truth (no flow, or one of three flow patterns at a given mean speed) the analyst fits
  diff  the diffusion-only model (alpha, r), and
  flow  the model with the true flow pattern and a free speed (alpha, r, vs),
to one observation type (tissue curves, depth profiles, 8 mm blocks or the native field), either
with exact boundary input and observations ("exact") or with the realistic errors of Q1 ("combo":
linearly interpolated, 15% low boundary input and CSF partial volume). Observations are whitened by
the noise standard deviation of iid node noise sigma = 0.02 mmol/L, so the noise-free squared
residual is the noncentrality of the chi^2 statistic: flow is detected when chi2_diff - chi2_flow
exceeds 3.84 (one extra parameter, 5% level).

Usage: python q2_flow_detect.py <job> | list     job = <truth>__<errors>__<model>__<obs>
Writes results/nd/q2/<job>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import q1_error_budget as q1  # noqa: E402
import twin  # noqa: E402

OUT = nd.ND_OUT / "q2"
SPEEDS = {"pial_inflow_drainage": (0.5, 1.0, 2.0, 4.0), "pial_to_ventricle_flux": (1.0, 4.0),
          "basal_to_dorsal": (1.0, 4.0)}
for F, vs in SPEEDS.items():
    for v in vs:
        q1.TRUTHS.setdefault(f"flow_{F}_{v:g}", dict(field="tensor", sD=1.0, r=0.018, flow=F, vs=v))
OBS = ("roi", "depth", "block8", "field")


def jobs():
    out = []
    for F, vs in SPEEDS.items():
        for v in vs:
            for err in ("exact", "combo"):
                for model in ("diff", f"flow-{F}"):
                    for o in OBS:
                        out.append(f"flow_{F}_{v:g}__{err}__{model}__{o}")
    for err in ("exact", "combo"):                       # no-flow truth: false-positive test for every pattern
        for o in OBS:
            out.append(f"base__{err}__diff__{o}")
            for F in SPEEDS:
                out.append(f"base__{err}__flow-{F}__{o}")
    return out


def run(job):
    truth, err, model, oname = job.split("__")
    S = nd.Setup()
    clock = twin.timer()
    c_true = q1.truth_trajectory(S, truth)
    bc_true = twin.Boundary(S.params)
    if err == "combo":
        Bt = np.stack([bc_true(t) for t in S.t])
        c_obs = nd.partial_volume(S, c_true, Bt, f0=0.5, ell=1.0)
        bc_model = nd.ScaledBoundary(nd.sampled_linear(bc_true), 0.85)
    else:
        c_obs, bc_model = c_true, bc_true
    K = S.mesh.stiffness(S.fields["DataTensor"])
    obs = nd.whitened_operators(S)[oname]
    tmask = S.tmask()
    if model == "diff":
        solver = nd.GenSolver(S.mesh, K, bc_model, S.t)
        y = obs(solver.weighted(c_obs))
        res = nd.fit(solver, obs, y, tmask, names=("sD", "r"), starts=((1.0, 0.02), (3.0, 0.15)))
    else:
        F = model.split("-", 1)[1]
        v = np.load(nd.ND_OUT / f"velocity_{F}.npy")
        L, q_in, _ = nd.advection_operator(S.mesh, v)
        solver = nd.GenSolver(S.mesh, K, bc_model, S.t, L=L, q_in=q_in)
        y = obs(solver.weighted(c_obs))
        prior = OUT / f"{truth}__{err}__diff__{oname}.json"
        a0, r0 = (1.0, 0.018)
        if prior.exists():
            d = json.load(open(prior))
            a0, r0 = d["fit"]["sD"], d["fit"]["r"]
        res = nd.fit(solver, obs, y, tmask, names=("sD", "r", "vs"), starts=((a0, r0, 1.0),), max_nfev=40)
    c_fit = res.pop("c")
    resid = obs(solver.weighted(c_fit))[tmask] - y[tmask]
    out = {"job": job, "truth": truth, "truth_spec": q1.TRUTHS[truth], "errors": err, "model": model, "obs": oname,
           "n_obs": int(obs.nrow * tmask.sum()), "chi2_noisefree": float(np.sum(resid ** 2)),
           "alpha_fit": res["sD"], "r_fit": res["r"], "vs_fit": res["vs"], "fit": res,
           **nd.score(S, c_fit, c_true, tmask), "seconds": round(clock(), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    print(json.dumps({k: out[k] for k in ("job", "chi2_noisefree", "alpha_fit", "r_fit", "vs_fit", "seconds")},
                     default=float))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(jobs()))
    elif sys.argv[1] == "truths":
        print("\n".join(sorted({j.split("__")[0] for j in jobs()})))
    else:
        run(sys.argv[1])
