"""B: honest uncertainty for the real-data enhancement factor.

1. Cluster-robust (sandwich) standard errors at the real-data fits. The residual of the native-field fit
   is spatially correlated over several mm, so treating 400,000 node values as independent understates
   the uncertainty. With J the sensitivities of the fitted observations to the free parameters and e the
   residual, V = (J'J)^-1 [sum_g (J_g' e_g)(J_g' e_g)'] (J'J)^-1, where the clusters g are cubic blocks of
   width h (8, 16, 32 mm) pooling all sessions. The design effect kappa = V_aa / V_iid_aa rescales the
   quasi-likelihood threshold of a profile (3.84 -> 3.84 kappa).
     se__<bc>__<obs>__<set>      set from q5 (base | amp | amp_pv | all); obs field | depth
2. Profiles with the CSF-side nuisances refitted: alpha fixed on a grid, (r, amp, f0) fitted.
     prof__<bc>__<obs>__<alpha>
Usage: python q7_uncertainty.py <job> | list | --summary      writes results/nd/q7u/<job>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ext_solver as ex  # noqa: E402
import nd_common as nd  # noqa: E402
import q4_real_data as q4  # noqa: E402
import twin  # noqa: E402

OUT = nd.ND_OUT / "q7u"
ALPHAS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0, 2.4, 2.8, 3.5, 4.5, 6.0)
BLOCKS = (8.0, 16.0, 32.0)
SET_NAMES = {"base": ("sD", "r"), "amp": ("sD", "r", "amp"), "amp_pv": ("sD", "r", "amp", "f0"),
             "all": ("sD", "r", "amp", "kx", "f0")}


def jobs():
    out = [f"se__{bc}__{o}__{s}" for bc in ("curve", "linear") for o in ("field", "depth") for s in SET_NAMES]
    out += [f"prof__{bc}__{o}__{a:g}" for bc in ("curve", "linear") for o in ("field", "depth") for a in ALPHAS]
    return out


def setup_real(bc):
    S = nd.Setup()
    G = nd.gonzo_data(S)
    c_obs = q4.real_on_clock(S, G)
    bc_model = twin.Boundary(S.params) if bc == "curve" else nd.LinearBoundary(G["times_h"], G["B_raw"])
    solver = ex.ExtSolver(S.mesh, S.mesh.stiffness(S.fields["DataTensor"]), bc_model, S.t)
    return S, c_obs, solver, ex.ExtModel(S, solver)


def cluster_ids(S, oname, h):
    """Cluster label of every observation row (one session): block of the node (field) or the row itself."""
    if oname == "field":
        ijk = np.floor((S.mesh.P - S.mesh.P.min(0)) / h).astype(int)
        return np.unique(ijk, axis=0, return_inverse=True)[1].ravel()
    return None


def run_se(bc, oname, sname):
    q5 = json.load(open(nd.ND_OUT / "q5" / f"real__{bc}__{oname}__{sname}.json"))
    p = {k: q5[k] for k in ("sD", "r", "amp", "kx", "f0")}
    S, c_obs, solver, model = setup_real(bc)
    names = [k for k in SET_NAMES[sname] if k not in q5["fit"]["active"]]          # free parameters only
    out = {"job": f"se__{bc}__{oname}__{sname}", "params": p,
           **sandwich(S, solver, model, dict(ex.DEFAULTS, **p), names, oname, S.tmask(), c_obs)}
    return out


def sandwich(S, solver, model, p, names, oname, tmask, c_obs):
    """iid and cluster-robust covariance of the free parameters names at p (field: cubic blocks of every
    width in BLOCKS pooled over sessions; otherwise each observation row is one cluster)."""
    o = nd.whitened_operators(S)[oname]
    m, dm = model(p, names)
    e = (o(solver.weighted(m)) - o(solver.weighted(c_obs)))[tmask]                  # (ns, nrow)
    J = np.stack([o(solver.weighted(dm[k]))[tmask] for k in names], -1)            # (ns, nrow, p)
    ns, nrow, npar = J.shape
    Jf, ef = J.reshape(-1, npar), e.ravel()
    H = Jf.T @ Jf
    Hi = np.linalg.inv(H)
    s2 = float(ef @ ef) / (len(ef) - npar)
    V_iid = s2 * Hi
    out = {"names": list(names), "n": int(len(ef)), "s2": s2,
           "se_iid": dict(zip(names, np.sqrt(np.diag(V_iid)).tolist())), "robust": {}}
    if oname == "field":
        for h in BLOCKS:
            g = np.tile(cluster_ids(S, oname, h), ns)
            ng = int(g.max()) + 1
            U = np.zeros((ng, npar))
            np.add.at(U, g, Jf * ef[:, None])                                           # per-cluster score
            V = Hi @ (U.T @ U) @ Hi
            out["robust"][f"{h:g}"] = {"clusters": ng, "se": dict(zip(names, np.sqrt(np.diag(V)).tolist())),
                                        "kappa": dict(zip(names, (np.diag(V) / np.diag(V_iid)).tolist()))}
    else:                                                                               # depth: rows of a tissue are one cluster
        U = (J * e[..., None]).sum(0)                                                   # per-row score over sessions
        V = Hi @ (U.T @ U) @ Hi
        out["robust"]["row"] = {"clusters": nrow, "se": dict(zip(names, np.sqrt(np.diag(V)).tolist())),
                                "kappa": dict(zip(names, (np.diag(V) / np.diag(V_iid)).tolist()))}
    return out


def run_prof(bc, oname, alpha):
    S, c_obs, solver, model = setup_real(bc)
    ops = nd.whitened_operators(S)
    tmask = S.tmask()
    y = ops[oname](solver.weighted(c_obs))
    names = ("r", "amp", "f0")
    starts = [{"r": 0.01, "amp": 0.7, "f0": 0.05}, {"r": 0.1, "amp": 1.0, "f0": 0.2}]
    res = ex.fit_ext(model, solver, ops[oname], y, tmask, names, starts, fixed={"sD": alpha})
    m = res.pop("m")
    return {"job": f"prof__{bc}__{oname}__{alpha:g}", "alpha": alpha, "fit": res,
            "chi2": ex.chi2_all(S, solver, m, c_obs, tmask, ops)}


def run(job):
    kind, bc, oname, x = job.split("__")
    clock = twin.timer()
    out = run_se(bc, oname, x) if kind == "se" else run_prof(bc, oname, float(x))
    out["seconds"] = round(clock(), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    print(json.dumps(out, default=float)[:800])


def summary():
    import pandas as pd
    rows = []
    for f in sorted(OUT.glob("se__*.json")):
        d = json.load(open(f))
        _, bc, o, s = d["job"].split("__")
        r = {"bc": bc, "obs": o, "set": s, "alpha": d["params"]["sD"], "se_iid": d["se_iid"].get("sD")}
        for h, v in d["robust"].items():
            r[f"se_{h}"] = v["se"].get("sD")
            r[f"kappa_{h}"] = v["kappa"].get("sD")
        rows.append(r)
    se = pd.DataFrame(rows)
    prow = []
    for f in sorted(OUT.glob("prof__*.json")):
        d = json.load(open(f))
        _, bc, o, a = d["job"].split("__")
        c = d["chi2"][o]
        prow.append({"bc": bc, "obs": o, "alpha": float(a), "chi2": c["chi2"], "n": c["n"], **d["fit"]["params"]})
    pr = pd.DataFrame(prow)
    lines = ["# B. Real-data uncertainty", "", "## Sandwich standard errors of alpha", "", se.round(4).to_markdown(index=False)]
    if len(pr):
        out = []
        for (bc, o), g in pr.groupby(["bc", "obs"]):
            g = g.sort_values("alpha")
            cmin = g.chi2.min()
            scaled = (g.chi2 - cmin) / (cmin / g.n.iloc[0])
            k = se[(se.bc == bc) & (se.obs == o) & (se.set == "amp_pv")]
            kap = float(k[[c for c in k.columns if c.startswith("kappa_")]].max(axis=1).iloc[0]) if len(k) else np.nan
            ok = g.alpha[scaled <= 3.84 * (kap if np.isfinite(kap) else 1.0)]
            out.append({"bc": bc, "obs": o, "alpha_min": float(g.alpha.iloc[int(np.argmin(g.chi2))]),
                        "kappa_max": kap, "not_excluded_lo": float(ok.min()), "not_excluded_hi": float(ok.max()),
                        "scaled_rise_at_3.5": float(scaled[np.isclose(g.alpha, 3.5)].iloc[0]) if np.any(np.isclose(g.alpha, 3.5)) else np.nan})
        lines += ["", "## Profiles with (r, amp, f0) refitted", "", pd.DataFrame(out).round(3).to_markdown(index=False),
                  "", pr.round(4).to_markdown(index=False)]
    open(nd.ND_OUT / "q7u_summary.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(jobs()))
    elif sys.argv[1] == "--summary":
        summary()
    else:
        run(sys.argv[1])
