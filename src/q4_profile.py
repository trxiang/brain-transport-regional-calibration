"""Q4 profile: how well do the real data constrain alpha?

For alpha fixed on a grid, the clearance r is fitted to the real Gonzo data (one observation type,
all sessions) and the misfit is recorded in every observation space. A flat profile chi^2(alpha)
means alpha is not identified by that observation type: alpha and r trade off along a ridge.

Usage: python q4_profile.py <job> | list     job = <field>__<bc>__<obs>__<alpha>
Writes results/nd/q4_profile/<job>.json; collect with --summary.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import q1_error_budget as q1  # noqa: E402
import q4_real_data as q4  # noqa: E402
import twin  # noqa: E402

OUT = twin.REPO / "results/nd/q4_profile"
ALPHAS = (0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)


def jobs():
    return [f"tensor__{bc}__{obs}__{a:g}" for bc in ("curve", "linear") for obs in ("roi", "depth", "field")
            for a in ALPHAS]


def run(job):
    field, bc, oname, a = job.split("__")
    alpha = float(a)
    S = nd.Setup()
    G = nd.gonzo_data(S)
    clock = twin.timer()
    c_obs = q4.real_on_clock(S, G)
    bc_model = twin.Boundary(S.params) if bc == "curve" else nd.LinearBoundary(G["times_h"], G["B_raw"])
    solver = nd.GenSolver(S.mesh, S.mesh.stiffness(q1.field_of(S, field)), bc_model, S.t)
    ops = nd.whitened_operators(S)
    tmask = S.tmask()
    y = ops[oname](solver.weighted(c_obs))
    res = nd.fit(solver, ops[oname], y, tmask, names=("r",), starts=((0.02,), (0.2,)), fixed={"sD": alpha},
                 max_nfev=40)
    c_fit = res.pop("c")
    chi2 = {}
    for k, o in ops.items():
        r_ = o(solver.weighted(c_fit))[tmask] - o(solver.weighted(c_obs))[tmask]
        chi2[k] = {"chi2": float(np.sum(r_ ** 2)), "n": int(o.nrow * tmask.sum())}
    out = {"job": job, "field": field, "bc": bc, "obs": oname, "alpha": alpha, "r_fit": res["r"], "fit": res,
           "chi2": chi2, "fit_roi_means": nd.roi_means(S, c_fit)[S.scan_idx].tolist(), "seconds": round(clock(), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    print(job, res["r"], {k: round(v["chi2"] / v["n"], 3) for k, v in chi2.items()}, out["seconds"])


def summary():
    import pandas as pd
    rows = []
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f))
        rows.append({"field": d["field"], "bc": d["bc"], "obs": d["obs"], "alpha": d["alpha"], "r_fit": d["r_fit"],
                     **{f"chi2_{k}": v["chi2"] for k, v in d["chi2"].items()}})
    df = pd.DataFrame(rows).sort_values(["bc", "obs", "alpha"])
    df.to_csv(twin.REPO / "results/nd/q4_profile.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(jobs()))
    elif sys.argv[1] == "--summary":
        summary()
    else:
        run(sys.argv[1])
