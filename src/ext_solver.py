"""Extended solver for the second round of studies (A: published pipeline, B: uncertainty, D: perfusion, E: groups).

Adds to nd_common.GenSolver (diffusion-only part):
  * a split diffusion operator  K0 + s_D K1  (s_D scales K1 only), e.g. alpha in the cerebral cortex only
    with alpha = 1 elsewhere, as in Vinje et al. (2023);
  * a fixed-concentration (Dirichlet) exterior, the observed concentration of the boundary voxel
    linearly interpolated in time (Vinje et al. 2023), in place of the Robin exchange with the CSF;
  * a distributed CSF-tissue exchange ("perfusion"): CSF enters the tissue at volumetric rate
    q(x) = Q qhat(x) (Q in mL/h, qhat = g w / sum g w, g = exp(-d_pial / ell_p)) carrying the CSF fluid
    concentration B_n/phi_SAS of the nearest pial node and returns carrying the interstitial fluid
    concentration c/phi_brain, i.e. a source  q (B_n/phi_SAS - c/phi_brain)  per node. This is the
    compartment picture of CSF perfusion of superficial tissue (Wahlin et al. 2026) and a minimal model of
    delivery along penetrating perivascular spaces (Causemann et al. 2026);
  * nuisance parameters amp (scale of the CSF input) and f0 (surface partial volume) as in q5_nuisance;
  * a generic bounded least-squares fit over any of sD, r, Q, kx, amp, f0.
Units: mm, h, mmol/L; Q in mL/h (1 mL = 1000 mm^3).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import nd_common as nd
import twin

ML = 1000.0                     # mm^3 per mL
VINJE_BOUNDS = {"sD": (1.0, 10.0), "r": (3.6e-4, 3.6)}            # alpha in [1, 10]; r in [1e-7, 1e-3] 1/s
BOUNDS = {"sD": (0.1, 10.0), "r": (0.0, 0.6), "Q": (0.0, 400.0), "kx": (0.05, 20.0), "amp": (0.3, 3.0),
          "f0": (0.0, 0.9)}
DEFAULTS = {"sD": 1.0, "r": 0.0, "Q": 0.0, "kx": 1.0, "amp": 1.0, "f0": 0.0}


def _cg(A):
    Dinv = 1.0 / A.diagonal()
    Mpre = sla.LinearOperator(A.shape, matvec=lambda x: Dinv * x)

    def solve(b, x0=None):
        try:
            x, info = sla.cg(A, b, x0=x0, M=Mpre, rtol=1e-11, atol=0.0, maxiter=10000)
        except TypeError:                                   # SciPy < 1.12
            x, info = sla.cg(A, b, x0=x0, M=Mpre, tol=1e-11, atol=0.0, maxiter=10000)
        if info != 0:
            raise RuntimeError(f"CG did not converge ({info})")
        return x
    return solve


class ExtSolver:
    """Backward Euler for  w dc/dt + (K0 + s_D K1) c + r w c + Robin + q c/phi_b = Robin source + q B_n/phi_SAS,
    or with c prescribed on the exterior nodes (dirichlet=(nodes, G), G (nt, nodes) on the clock)."""

    def __init__(self, mesh, K1, bc, times_h, phys=twin.Physics(), K0=None, perf=None, dirichlet=None):
        self.mesh, self.K1, self.t, self.phys = mesh, K1.tocsr(), np.asarray(times_h, float), phys
        self.K0 = sp.csr_matrix(K1.shape) if K0 is None else K0.tocsr()
        self.Bt = np.stack([bc(t) for t in self.t]) if bc is not None else np.zeros((len(self.t), mesh.n))
        self.perf = perf                                    # dict(qhat=(n,), near=(n,) nearest pial node)
        if perf is not None:
            self.Bn = self.Bt[:, perf["near"]]
        self.dir = None
        if dirichlet is not None:
            nodes, G = dirichlet
            free = np.ones(mesh.n, bool)
            free[nodes] = False
            self.dir = {"nodes": np.asarray(nodes), "G": np.asarray(G, float), "free": np.flatnonzero(free)}
            self.K1_FD = self.K1[self.dir["free"]][:, self.dir["nodes"]]
            self.K0_FD = self.K0[self.dir["free"]][:, self.dir["nodes"]]
        self._cache = {}

    def _factor(self, sD, r, Q, kx, dt):
        key = (round(sD, 14), round(r, 14), round(Q, 12), round(kx, 14), round(dt, 12))
        if key not in self._cache:
            m, ph = self.mesh, self.phys
            diag = m.w * (1.0 + r * dt)
            if self.dir is None:
                diag = diag + dt * m.a * (kx * ph.k) / ph.phi_brain
            if self.perf is not None and Q != 0.0:
                diag = diag + dt * ML * Q * self.perf["qhat"] / ph.phi_brain
            A = (sp.diags(diag) + dt * (self.K0 + sD * self.K1)).tocsr()
            if self.dir is not None:
                F = self.dir["free"]
                A = A[F][:, F].tocsr()
            self._cache[key] = _cg(A)
            if len(self._cache) > 14:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def run(self, sD: float, r: float, vs: float = 0.0, sens=(), Q: float = 0.0, kx: float = 1.0):
        """c (nt,n) and sensitivities for the names in sens among 'sD', 'r', 'Q', 'kx' (vs must be 0)."""
        assert vs == 0.0
        m, ph = self.mesh, self.phys
        nt = len(self.t)
        c = np.zeros((nt, m.n))
        sens = [k for k in sens if k in ("sD", "r", "Q", "kx")]
        S = {k: np.zeros((nt, m.n)) for k in sens}
        F = None if self.dir is None else self.dir["free"]
        for i in range(1, nt):
            dt = self.t[i] - self.t[i - 1]
            solve = self._factor(sD, r, Q, kx, dt)
            rhs = m.w * c[i - 1]
            if self.dir is None:
                rhs = rhs + dt * m.a * (kx * ph.k) * self.Bt[i] / ph.phi_sas
            if self.perf is not None and Q != 0.0:
                rhs = rhs + dt * ML * Q * self.perf["qhat"] * self.Bn[i] / ph.phi_sas
            if F is None:
                c[i] = solve(rhs, c[i - 1])
            else:
                g = self.dir["G"][i]
                c[i, self.dir["nodes"]] = g
                c[i, F] = solve(rhs[F] - dt * (self.K0_FD @ g + sD * (self.K1_FD @ g)), c[i - 1, F])
            for k in sens:
                if k == "sD":
                    b = m.w * S[k][i - 1] - dt * (self.K1 @ c[i])
                elif k == "r":
                    b = m.w * S[k][i - 1] - dt * m.w * c[i]
                elif k == "Q":
                    b = m.w * S[k][i - 1] + dt * ML * self.perf["qhat"] * (self.Bn[i] / ph.phi_sas - c[i] / ph.phi_brain)
                else:                                      # kx
                    b = m.w * S[k][i - 1] + dt * m.a * ph.k * (self.Bt[i] / ph.phi_sas - c[i] / ph.phi_brain)
                if F is None:
                    S[k][i] = solve(b, S[k][i - 1])
                else:
                    S[k][i, F] = solve(b[F], S[k][i - 1, F])
        return c, S

    def weighted(self, c):
        return c * np.sqrt(self.mesh.w / self.mesh.V)[None, :]


# ----------------------------------------------------------------------------- building blocks
def regional_isotropic(S) -> np.ndarray:
    """Group-averaged, regionally constant isotropic diffusivity (Vinje et al. 2023): the volume mean of
    the DTI mean diffusivity in each tissue label, per cell."""
    m, iso = S.mesh, S.fields["DataIsotropic"]
    out = np.zeros(len(m.C))
    for j in np.unique(m.label):
        sel = m.label == j
        out[sel] = np.average(iso[sel], weights=m.vol[sel])
    return out


def split_stiffness(S, D, alpha_label=None):
    """(K0, K1): K1 holds the cells whose diffusivity alpha multiplies (all, or label alpha_label)."""
    if alpha_label is None:
        return None, S.mesh.stiffness(D)
    sel = S.mesh.label == alpha_label
    D1 = D * (sel if D.ndim == 1 else sel[:, None, None])
    return S.mesh.stiffness(D - D1), S.mesh.stiffness(D1)


def perfusion_weights(S, ell_p: float) -> dict:
    """qhat (sums to 1) decaying with distance from the pial (outer, non-ventricular) surface, and the
    nearest pial node of every node (whose CSF concentration enters)."""
    d, j = cKDTree(S.mesh.P[S.outer_nodes]).query(S.mesh.P)
    g = np.exp(-d / ell_p) * S.mesh.w
    return {"qhat": g / g.sum(), "near": S.outer_nodes[j], "d_pial": d}


def dirichlet_from_obs(S, c_obs, max_h=None):
    """Exterior-node values of the observations at the scans (zero at injection), linearly interpolated on
    the clock and held after the last scan used: the boundary data of Vinje et al. (2023)."""
    idx = [k for k, th in zip(S.scan_idx, nd.SCAN_H) if max_h is None or th <= max_h + 1e-9]
    th = np.concatenate([[0.0], S.t[idx]])
    vals = np.vstack([np.zeros((1, len(S.bnodes))), c_obs[idx][:, S.bnodes]])
    lb = nd.LinearBoundary(th, vals)
    return S.bnodes, np.stack([lb(t) for t in S.t])


class ExtModel:
    """Observed-model prediction m = (1 - f) amp c + f B_n (f = f0 exp(-d/ell)) and its sensitivities."""

    def __init__(self, S, solver: ExtSolver, ell: float = 1.0):
        self.solver = solver
        self.g = np.exp(-S.dist / ell)[None, :]
        self.Bn = solver.Bt[:, S.nearest_b]

    def __call__(self, p, names):
        sens = tuple(k for k in names if k in ("sD", "r", "Q", "kx"))
        c1, S1 = self.solver.run(p["sD"], p["r"], sens=sens, Q=p["Q"], kx=p["kx"])
        f = p["f0"] * self.g
        m = (1.0 - f) * p["amp"] * c1 + f * self.Bn
        dm = {k: (1.0 - f) * p["amp"] * S1[k] for k in sens}
        if "amp" in names:
            dm["amp"] = (1.0 - f) * c1
        if "f0" in names:
            dm["f0"] = self.g * (self.Bn - p["amp"] * c1)
        return m, dm


def fit_ext(model, solver, obs, y, tmask, names, starts, fixed=None, bounds=None, max_nfev=80):
    """Bounded least squares of obs(m) against y over tmask, with exact sensitivities, from every start."""
    bnd = dict(BOUNDS, **(bounds or {}))
    base = dict(DEFAULTS, **(fixed or {}))
    lb = np.array([bnd[k][0] for k in names])
    ub = np.array([bnd[k][1] for k in names])
    spans = ub - lb
    yv = y[tmask]
    norm = np.linalg.norm(yv)
    cache = {}

    def evaluate(u):
        key = tuple(np.round(u, 15))
        if key not in cache:
            p = dict(base, **dict(zip(names, lb + u * spans)))
            m, dm = model(p, names)
            res = (obs(solver.weighted(m))[tmask] - yv).ravel() / norm
            J = np.stack([obs(solver.weighted(dm[k]))[tmask].ravel() / norm * spans[j] for j, k in enumerate(names)], 1)
            cache.clear()
            cache[key] = (res, J, m)
        return cache[key]

    best = None
    for x0 in starts:
        u0 = np.clip((np.array([x0.get(k, base[k]) for k in names]) - lb) / spans, 0.0, 1.0)
        sol = least_squares(lambda u: evaluate(u)[0], u0, jac=lambda u: evaluate(u)[1], bounds=(0.0, 1.0),
                            method="trf", xtol=1e-12, ftol=1e-14, gtol=1e-12, max_nfev=max_nfev)
        res, J, m = evaluate(sol.x)
        cost = float(np.linalg.norm(res))
        if best is None or cost < best["observed_L2"]:
            p = dict(base, **dict(zip(names, (lb + sol.x * spans).tolist())))
            best = {"params": p, "observed_L2": cost, "m": m, "nfev": int(sol.nfev), "status": int(sol.status),
                    "active": [k for k, a in zip(names, sol.active_mask) if a != 0], "start": x0}
    return best


def chi2_all(S, solver, m, c_obs, tmask, ops=None):
    """Misfit of a prediction in every whitened observation space (chi^2 and its count)."""
    ops = ops or nd.whitened_operators(S)
    out = {}
    for k, o in ops.items():
        r_ = o(solver.weighted(m))[tmask] - o(solver.weighted(c_obs))[tmask]
        out[k] = {"chi2": float(np.sum(r_ ** 2)), "n": int(o.nrow * tmask.sum())}
    return out
