"""D: CSF perfusion of superficial tissue (perivascular delivery) as a third transport hypothesis.

Model (ext_solver): CSF enters the tissue at volumetric rate q(x) = Q qhat(x), qhat ~ exp(-d_pial / ell_p),
carrying the CSF fluid concentration of the nearest pial node, and returns carrying the interstitial
concentration (source q (B_n/phi_SAS - c/phi_brain)). Q in mL/h; Wahlin et al. (2026, PNAS) estimate
CSF-to-ISF inflow of 45 +- 20 mL/h, mostly cortical (about 4 um/min across the Gonzo pial surface).

Truths perf_Q<Q>_l<ell>: tensor field, alpha 1, r 0.018, curve input, Q in {15, 45} mL/h, ell_p in {1, 3} mm.
Synthetic jobs  syn__<truth>__<err>__<model>__<obs>
    err    exact | combo (Q1 combined errors)
    model  diff (alpha, r) | perf (Q, r; alpha 1; ell_p of the truth, 1 mm for base) | both (alpha, r, Q)
Real jobs       real__<bc>__<obs>__<model>__<ell>__<nuis>     model perf | both; nuis none | amp
Usage: python q8_perfusion.py <job> | list | truths <truth> | --summary
Writes results/nd/q8/<job>.json; truths cached in results/nd/truths/.
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
import twin  # noqa: E402

OUT = nd.ND_OUT / "q8"
PERF_TRUTHS = {f"perf_Q{Q:g}_l{l:g}": (Q, l) for Q in (15.0, 45.0) for l in (1.0, 3.0)}
MODELS = {"diff": ("sD", "r"), "perf": ("Q", "r"), "both": ("sD", "r", "Q")}


def jobs():
    out = []
    for t in PERF_TRUTHS:
        out += [f"syn__{t}__{e}__diff__{o}" for e in ("exact", "combo") for o in ("roi", "depth", "field")]
        out += [f"syn__{t}__exact__perf__field"]
        out += [f"syn__{t}__combo__perf__{o}" for o in ("roi", "depth", "field")]
        out += [f"syn__{t}__combo__both__{o}" for o in ("depth", "field")]
    out += [f"syn__base__combo__{m}__{o}" for m in ("perf", "both") for o in ("roi", "depth", "field")]
    out += [f"real__{bc}__{o}__{m}__{l:g}__{n}" for bc in ("curve", "linear") for o in ("roi", "depth", "field")
            for m in ("perf", "both") for l in (1.0, 3.0) for n in ("none", "amp")]
    return out


def truth(S, key):
    if key in q1.TRUTHS:
        return q1.truth_trajectory(S, key)
    f = q1.TRUTH_CACHE / f"{key}.npy"
    if f.exists():
        return np.load(f).astype(float)
    Q, ell = PERF_TRUTHS[key]
    sol = ex.ExtSolver(S.mesh, S.mesh.stiffness(S.fields["DataTensor"]), twin.Boundary(S.params), S.t,
                       perf=ex.perfusion_weights(S, ell))
    c, _ = sol.run(1.0, 0.018, Q=Q)
    q1.TRUTH_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, c.astype(np.float32))
    return c


def run(job):
    parts = job.split("__")
    S = nd.Setup()
    clock = twin.timer()
    bc_true = twin.Boundary(S.params)
    ops = nd.whitened_operators(S)
    tmask = S.tmask()
    names_extra, fixed, true_spec = (), {}, None
    if parts[0] == "syn":
        _, tkey, err, mname, oname = parts
        c_true = truth(S, tkey)
        c_obs, bc_model = c_true, bc_true
        if err == "combo":
            Bt = np.stack([bc_true(t) for t in S.t])
            c_obs = nd.add_noise(nd.partial_volume(S, c_true, Bt, f0=0.5, ell=1.0), q1.NOISE_SIGMA, seed=20261101)
            bc_model = nd.ScaledBoundary(nd.sampled_linear(bc_true), 0.85)
        ell = PERF_TRUTHS.get(tkey, (0.0, 1.0))[1]
        true_spec = {"Q": PERF_TRUTHS.get(tkey, (0.0, 1.0))[0], "ell_p": ell, "alpha": 1.0, "r": 0.018}
        nuis = "none"
    else:
        _, bc, oname, mname, ell, nuis = parts
        ell = float(ell)
        G = nd.gonzo_data(S)
        c_obs = q4.real_on_clock(S, G)
        bc_model = bc_true if bc == "curve" else nd.LinearBoundary(G["times_h"], G["B_raw"])
        if nuis == "amp":
            names_extra = ("amp",)
    pw = ex.perfusion_weights(S, ell) if mname != "diff" else None
    solver = ex.ExtSolver(S.mesh, S.mesh.stiffness(S.fields["DataTensor"]), bc_model, S.t, perf=pw)
    model = ex.ExtModel(S, solver)
    names = MODELS[mname] + names_extra
    y = ops[oname](solver.weighted(c_obs))
    starts = [{"sD": 1.0, "r": 0.02, "Q": 20.0, "amp": 0.8}, {"sD": 2.5, "r": 0.1, "Q": 80.0, "amp": 1.0}]
    res = ex.fit_ext(model, solver, ops[oname], y, tmask, names, starts, fixed=fixed)
    m = res.pop("m")
    out = {"job": job, "parts": parts, "model": mname, "obs": oname, "ell_p": ell, "nuis": nuis, "truth": true_spec,
           **res["params"], "fit": res, "chi2": ex.chi2_all(S, solver, m, c_obs, tmask, ops),
           "seconds": round(clock(), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    print(job, {k: round(res["params"][k], 4) for k in ("sD", "r", "Q", "amp")}, res["active"],
          {k: round(v["chi2"] / v["n"], 3) for k, v in out["chi2"].items()}, out["seconds"])


def summary():
    import pandas as pd
    rows = []
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f))
        rows.append({"job": d["job"], "model": d["model"], "obs": d["obs"], "ell_p": d["ell_p"], "nuis": d["nuis"],
                     **{k: d[k] for k in ("sD", "r", "Q", "amp")}, "active": ",".join(d["fit"]["active"]),
                     **{f"chi2_{k}": v["chi2"] for k, v in d["chi2"].items()},
                     **{f"chi2n_{k}": v["chi2"] / v["n"] for k, v in d["chi2"].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(nd.ND_OUT / "q8_summary.csv", index=False)
    open(nd.ND_OUT / "q8_summary.md", "w").write(df.round(4).to_markdown(index=False))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(jobs()))
    elif sys.argv[1] == "truths":
        truth(nd.Setup(), sys.argv[2])
    elif sys.argv[1] == "--summary":
        summary()
    else:
        run(sys.argv[1])
