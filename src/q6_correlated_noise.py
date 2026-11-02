"""Q6: does flow detection survive spatially correlated noise?

Q2 whitens observations assuming independent node noise (sigma = 0.02 mmol/L), under which a region
mean over ~30,000 nodes is known to ~1e-4 mmol/L. Here the node noise has the same SD but a fraction
1 - eta of its variance is a smooth Gaussian random field with correlation exp(-d^2 / (2 ell^2)):
    Sigma = sigma^2 [ (1 - eta) R_ell + eta I ].
Observation covariances A Sigma A^T (tissue curves, depth profiles, 8 mm blocks; one session at a
time, sessions independent) are built with random Fourier features (M features, unbiased for R_ell).
The diffusion-only model is refitted by generalised least squares (whitening by the Cholesky factor),
so its squared residual is the noncentrality of the flow likelihood-ratio test (the flow model with
the true pattern fits these exact-input truths exactly).

Usage: python q6_correlated_noise.py <job> | list | --summary | --realcorr
  job = <truth>__<obs>__<ell>          obs = roi | depth | block8, ell in mm (eta = 0.5)
Writes results/nd/q6/<job>.json; --realcorr writes the correlogram of the real-data residual.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sl

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import q1_error_budget as q1  # noqa: E402
import twin  # noqa: E402

ND = twin.REPO / "results/nd"
OUT = ND / "q6"
SIGMA, ETA, M_FEATURES = 0.02, 0.5, 20000
TRUTHS = ("flow_pial_inflow_drainage_0.5", "flow_pial_inflow_drainage_1", "flow_pial_to_ventricle_flux_1",
          "flow_basal_to_dorsal_1")
ELLS = (2.5, 5.0, 10.0)


def jobs():
    return [f"{t}__{o}__{e:g}" for t in TRUTHS for o in ("roi", "depth", "block8") for e in ELLS]


def mean_rows(S, oname):
    """(m,n) rows giving each observation as a weighted mean of node values."""
    if oname == "roi":
        return S.roi.mean_op.toarray()
    if oname == "depth":
        D = nd.depth_tissue_rows(S)
        return D / D.sum(1, keepdims=True)
    B = nd.block_rows(S, 8.0)
    return (nd.sp.diags(1.0 / np.asarray(B.sum(1)).ravel()) @ B).tocsr()


def obs_covariance(S, A, ell, eta=ETA, seed=20261102):
    """A Sigma A^T for one session, random Fourier features for the smooth part."""
    rng = np.random.default_rng(seed)
    m = A.shape[0]
    G = np.zeros((m, m))
    for start in range(0, M_FEATURES, 1000):
        k = min(1000, M_FEATURES - start)
        Om = rng.normal(0.0, 1.0 / ell, (3, k))
        b = rng.uniform(0.0, 2 * np.pi, k)
        Phi = np.sqrt(2.0 / M_FEATURES) * np.cos(S.mesh.P @ Om + b)           # (n,k)
        AP = A @ Phi
        G += AP @ AP.T
    AAt = (A @ A.T).toarray() if hasattr(A, "toarray") else A @ A.T
    if hasattr(AAt, "toarray"):
        AAt = AAt.toarray()
    return SIGMA ** 2 * ((1.0 - eta) * G + eta * np.asarray(AAt))


class GLSObs:
    """Observation operator on weighted coordinates: L^{-1} A c (whitened by the correlated covariance)."""

    def __init__(self, S, A, C):
        self.A, self.sw = A, np.sqrt(S.mesh.w / S.mesh.V)
        self.L = np.linalg.cholesky(C)
        self.nrow = A.shape[0]

    def __call__(self, z):                                  # z (nt,n) weighted -> (nt,m)
        c = z / self.sw[None, :]
        means = np.asarray((self.A @ c.T)).T
        return sl.solve_triangular(self.L, means.T, lower=True).T


def run(job):
    truth, oname, ell = job.split("__")
    ell = float(ell)
    S = nd.Setup()
    clock = twin.timer()
    A = mean_rows(S, oname)
    c_true = q1.truth_trajectory(S, truth)
    solver = nd.GenSolver(S.mesh, S.mesh.stiffness(S.fields["DataTensor"]), twin.Boundary(S.params), S.t)
    tmask = S.tmask()
    fits = {}
    for eta in (1.0, ETA):                    # iid with full covariance (shared interface nodes), then correlated
        C = obs_covariance(S, A, ell, eta)
        obs = GLSObs(S, A, C)
        y = obs(solver.weighted(c_true))
        res = nd.fit(solver, obs, y, tmask, names=("sD", "r"), starts=((1.0, 0.02), (3.0, 0.15)))
        c_fit = res.pop("c")
        fits[eta] = {"alpha": res["sD"], "r": res["r"], "C": C,
                     "lambda": float(np.sum((obs(solver.weighted(c_fit)) - y)[tmask] ** 2))}
    iid = json.load(open(ND / "q2" / f"{truth}__exact__diff__{oname}.json"))
    out = {"job": job, "truth": truth, "obs": oname, "ell": ell, "eta": ETA, "alpha_fit": fits[ETA]["alpha"],
           "r_fit": fits[ETA]["r"], "lambda_corr": fits[ETA]["lambda"], "lambda_iid": fits[1.0]["lambda"],
           "lambda_iid_diag_q2": iid["chi2_noisefree"], "obs_sd_corr": np.sqrt(np.diag(fits[ETA]["C"])).tolist(),
           "seconds": round(clock(), 1)}
    lam = out["lambda_corr"]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{job}.json", "w"), indent=2, default=float)
    print(job, "lambda corr", lam, "iid", out["lambda_iid"], out["seconds"])


def realcorr():
    """Correlogram of the real-data residual (data minus the native-field diffusion fit) per session."""
    S = nd.Setup()
    G = nd.gonzo_data(S)
    fit = np.load(ND / "q4" / "tensor__curve__field__all__diff.npz")["c_fit_sessions"].astype(float)
    R = G["c"][1:] - fit
    rng = np.random.default_rng(0)
    n = S.mesh.n
    i, j = rng.integers(0, n, 400000), rng.integers(0, n, 400000)
    d = np.linalg.norm(S.mesh.P[i] - S.mesh.P[j], axis=1)
    edges = np.array([0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 60])
    out = {"edges_mm": edges.tolist(), "sessions": []}
    # close pairs are rare among random pairs: add neighbours within 20 mm
    from scipy.spatial import cKDTree
    pairs = cKDTree(S.mesh.P).query_pairs(20.0, output_type="ndarray")
    sel = rng.choice(len(pairs), min(len(pairs), 2000000), replace=False)
    i2, j2 = pairs[sel, 0], pairs[sel, 1]
    I, J = np.concatenate([i, i2]), np.concatenate([j, j2])
    Dd = np.linalg.norm(S.mesh.P[I] - S.mesh.P[J], axis=1)
    for s in range(R.shape[0]):
        r = R[s] - R[s].mean()
        v = r.var()
        rho = [float(np.mean(r[I[(Dd >= a) & (Dd < b)]] * r[J[(Dd >= a) & (Dd < b)]]) / v)
               if np.any((Dd >= a) & (Dd < b)) else None for a, b in zip(edges[:-1], edges[1:])]
        out["sessions"].append({"t_h": float(nd.SCAN_H[s]), "sd": float(np.sqrt(v)), "rho": rho})
        print(f"{nd.SCAN_H[s]:.1f} h  sd {np.sqrt(v):.4f}  rho", [None if x is None else round(x, 2) for x in rho])
    json.dump(out, open(ND / "real_residual_correlogram.json", "w"), indent=1)


def summary():
    import pandas as pd
    from scipy.stats import chi2, ncx2
    crit = chi2.ppf(0.95, 1)
    rows = []
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f))
        rows.append({"truth": d["truth"], "obs": d["obs"], "ell_mm": d["ell"], "lambda_iid": d["lambda_iid"],
                     "lambda_corr": d["lambda_corr"],
                     "power_iid": 1 - ncx2.cdf(crit, 1, max(d["lambda_iid"], 1e-12)),
                     "power_corr": 1 - ncx2.cdf(crit, 1, max(d["lambda_corr"], 1e-12)),
                     "alpha_gls": d["alpha_fit"], "mean_obs_sd": float(np.mean(d["obs_sd_corr"]))})
    df = pd.DataFrame(rows).sort_values(["truth", "obs", "ell_mm"])
    df.to_csv(ND / "q6_summary.csv", index=False)
    open(ND / "q6_summary.md", "w").write(df.round(4).to_markdown(index=False))
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "list":
        print("\n".join(jobs()))
    elif a == "--summary":
        summary()
    elif a == "--realcorr":
        realcorr()
    else:
        run(a)
