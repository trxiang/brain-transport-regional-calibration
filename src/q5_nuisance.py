"""Q5: fit the CSF-side unknowns as nuisance parameters alongside (alpha, r).

The Q1 error budget and the real-data fingerprint point at the CSF input and partial volume. Here the
analyst model gains, in any combination,
  amp   scale of the CSF input reaching the tissue          c_model = amp * c(B_measured)   (c is linear in B)
  kx    multiplier of the exchange coefficient k             Robin flux k kx (c/phi_b - B/phi_s)
  f0    partial-volume fraction near the surface (ell fixed) m = (1 - f) c_model + f B_nearest,
                                                             f = f0 exp(-d / ell)
and alpha, r are refitted. Two uses:
  synthetic  truth (base alpha 1 | vinje alpha 3.5) observed with the Q1 combined errors (linear CSF
             input 15% low, partial volume f0 0.5, ell 1 mm): does the nuisance fit remove the bias?
  real       the Gonzo maps: do the nuisances explain the misfit, and where does alpha go?

Usage: python q5_nuisance.py <job> | list
  job = syn__<truth>__<obs>__<set>   or   real__<bc>__<obs>__<set>
  set = base | amp | kx | pv | pv2 | amp_pv | amp_kx | all        (pv2: ell = 2 mm)
Writes results/nd/q5/<job>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import q1_error_budget as q1  # noqa: E402
import q4_real_data as q4  # noqa: E402
import twin  # noqa: E402

OUT = nd.ND_OUT / "q5"
SETS = {"base": ("sD", "r"), "amp": ("sD", "r", "amp"), "kx": ("sD", "r", "kx"), "pv": ("sD", "r", "f0"),
        "pv2": ("sD", "r", "f0"), "amp_pv": ("sD", "r", "amp", "f0"), "amp_kx": ("sD", "r", "amp", "kx"),
        "all": ("sD", "r", "amp", "kx", "f0")}
BOUNDS = {"sD": (0.1, 10.0), "r": (0.0, 0.6), "amp": (0.3, 3.0), "kx": (0.05, 20.0), "f0": (0.0, 0.9)}
DEFAULTS = {"sD": 1.0, "r": 0.0, "amp": 1.0, "kx": 1.0, "f0": 0.0}


def jobs():
    out = [f"syn__{t}__{o}__{s}" for t in ("base", "vinje") for o in ("roi", "depth", "field")
           for s in ("base", "amp_pv", "all")]
    out += [f"real__{bc}__{o}__{s}" for bc in ("curve", "linear") for o in ("roi", "depth", "field") for s in SETS]
    return out


class NuisanceModel:
    """Model observations m(p) and their parameter sensitivities on the solver clock."""

    def __init__(self, S, solver, ell):
        self.solver = solver
        self.g = np.exp(-S.dist / ell)[None, :]
        self.Bn = solver.Bt[:, S.nearest_b]

    def __call__(self, p, names):
        sens = tuple(k for k in names if k in ("sD", "r", "kx"))
        c1, S1 = self.solver.run(p["sD"], p["r"], sens=sens, kx=p["kx"])
        f = p["f0"] * self.g
        m = (1.0 - f) * p["amp"] * c1 + f * self.Bn
        dm = {k: (1.0 - f) * p["amp"] * S1[k] for k in sens}
        if "amp" in names:
            dm["amp"] = (1.0 - f) * c1
        if "f0" in names:
            dm["f0"] = self.g * (self.Bn - p["amp"] * c1)
        return m, dm


def fit_nuisance(model, solver, obs, y, tmask, names, starts, max_nfev=80):
    lb = np.array([BOUNDS[k][0] for k in names])
    ub = np.array([BOUNDS[k][1] for k in names])
    spans = ub - lb
    yv = y[tmask]
    norm = np.linalg.norm(yv)
    cache = {}

    def evaluate(u):
        key = tuple(np.round(u, 15))
        if key not in cache:
            p = dict(DEFAULTS, **dict(zip(names, lb + u * spans)))
            m, dm = model(p, names)
            res = (obs(solver.weighted(m))[tmask] - yv).ravel() / norm
            J = np.stack([obs(solver.weighted(dm[k]))[tmask].ravel() / norm * spans[j] for j, k in enumerate(names)], 1)
            cache.clear()
            cache[key] = (res, J, m)
        return cache[key]

    best = None
    for x0 in starts:
        u0 = np.clip((np.array([x0.get(k, DEFAULTS[k]) for k in names]) - lb) / spans, 0.0, 1.0)
        sol = least_squares(lambda u: evaluate(u)[0], u0, jac=lambda u: evaluate(u)[1], bounds=(0.0, 1.0),
                            method="trf", xtol=1e-12, ftol=1e-14, gtol=1e-12, max_nfev=max_nfev)
        res, J, m = evaluate(sol.x)
        cost = float(np.linalg.norm(res))
        if best is None or cost < best["observed_L2"]:
            p = dict(DEFAULTS, **dict(zip(names, (lb + sol.x * spans).tolist())))
            best = {"params": p, "observed_L2": cost, "m": m, "nfev": int(sol.nfev), "status": int(sol.status),
                    "active": [k for k, a in zip(names, sol.active_mask) if a != 0], "start": x0}
    return best


def run(job):
    kind, a, oname, sname = job.split("__")
    S = nd.Setup()
    clock = twin.timer()
    names = SETS[sname]
    ell = 2.0 if sname == "pv2" else 1.0
    K = S.mesh.stiffness(S.fields["DataTensor"])
    ops = nd.whitened_operators(S)
    tmask = S.tmask()
    truth = None
    if kind == "syn":
        truth = q1.TRUTHS[a]
        bc_true = twin.Boundary(S.params)
        Bt = np.stack([bc_true(t) for t in S.t])
        c_obs = nd.partial_volume(S, q1.truth_trajectory(S, a), Bt, f0=0.5, ell=1.0)
        bc_model = nd.ScaledBoundary(nd.sampled_linear(bc_true), 0.85)     # the analyst's measured input
    else:
        G = nd.gonzo_data(S)
        c_obs = q4.real_on_clock(S, G)
        bc_model = twin.Boundary(S.params) if a == "curve" else nd.LinearBoundary(G["times_h"], G["B_raw"])
    solver = nd.GenSolver(S.mesh, K, bc_model, S.t)
    model = NuisanceModel(S, solver, ell)
    y = ops[oname](solver.weighted(c_obs))
    starts = [{"sD": 1.0, "r": 0.02}, {"sD": 3.0, "r": 0.15, "amp": 0.8, "kx": 0.5, "f0": 0.3}]
    res = fit_nuisance(model, solver, ops[oname], y, tmask, names, starts)
    m = res.pop("m")
    chi2 = {}
    for k, o in ops.items():
        r_ = o(solver.weighted(m))[tmask] - o(solver.weighted(c_obs))[tmask]
        chi2[k] = {"chi2": float(np.sum(r_ ** 2)), "n": int(o.nrow * tmask.sum())}
    out = {"job": job, "kind": kind, "case": a, "obs": oname, "set": sname, "ell": ell, **res["params"],
           "fit": res, "chi2": chi2, "truth": truth, "seconds": round(clock(), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    print(job, {k: round(v, 4) for k, v in res["params"].items()}, res["active"],
          {k: round(v["chi2"] / v["n"], 3) for k, v in chi2.items()}, out["seconds"])


def summary():
    import pandas as pd
    rows = []
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f))
        rows.append({"kind": d["kind"], "case": d["case"], "obs": d["obs"], "set": d["set"],
                     **{k: d[k] for k in ("sD", "r", "amp", "kx", "f0")}, "active": ",".join(d["fit"]["active"]),
                     **{f"chi2n_{k}": v["chi2"] / v["n"] for k, v in d["chi2"].items()}})
    df = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(SETS)}
    df = df.sort_values(["kind", "case", "obs", "set"], key=lambda c: c.map(order) if c.name == "set" else c)
    df.to_csv(nd.ND_OUT / "q5_summary.csv", index=False)
    with open(nd.ND_OUT / "q5_summary.md", "w") as fh:
        fh.write(df.round(4).to_markdown(index=False))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(jobs()))
    elif sys.argv[1] == "--summary":
        summary()
    else:
        run(sys.argv[1])
