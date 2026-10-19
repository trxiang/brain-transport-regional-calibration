"""Python re-implementation ("twin") of the official DuMux braindiffusion miniapp.

Model (dumux_miniapp/app/main.cc, model.hh), in mm and hours:

    dc/dt = div(s_D D(x) grad c) - r c                      in the brain
    outward flux  j_n = k (c/phi_brain - B(t)/phi_SAS)       on the exterior surface

Discretisation matching the DuMux box scheme on tetrahedra:
  * P1 stiffness  K_ij = sum_T |T| grad(phi_i) . D_T grad(phi_j)
  * lumped storage and sink on the barycentric dual volume w_i = sum_T |T|/4
  * Robin term lumped on boundary dual area a_i = sum_F |F|/3, evaluated at the new time
  * backward Euler on the official 75-state clock, B(t) evaluated at the new time
  * B(t) = a * tau * exp(-b * tau), tau in days, fitted per boundary vertex to the
    five public checkpoints (Levenberg-Marquardt from (0.2, 1.0), as the miniapp does)

Signed values are never clipped.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

REPO = Path(__file__).resolve().parents[1]
CORE = Path(os.environ.get("BT_DATA", REPO / "data"))                       # data package (03_RESULTS, 04_DATA)
EXT = Path(os.environ.get("BT_DATA_EXTENSION", REPO / "data_extension"))    # archived trajectories (Step 0, C1, C2)

CHECKPOINT_S = np.array([0.0, 15716.0, 86001.0, 172280.0, 251641.0])
K_SURF = 2e-4 * 3600.0       # surface conductance, mm/h
PHI_BRAIN = 0.2
PHI_SAS = 0.4


# ----------------------------------------------------------------------------- mesh
@dataclass
class Mesh:
    P: np.ndarray                 # (n,3) node coordinates, mm
    C: np.ndarray                 # (m,4) tetrahedra
    label: np.ndarray             # (m,) tissue label 1 cortical GM, 2 WM, 3 subcortical GM
    vol: np.ndarray = field(init=False)
    G: np.ndarray = field(init=False)      # (m,4,3) gradients of barycentric functions
    w: np.ndarray = field(init=False)      # (n,) dual volumes
    a: np.ndarray = field(init=False)      # (n,) boundary dual areas
    ext: np.ndarray = field(init=False)    # exterior triangles

    def __post_init__(self):
        X = self.P[self.C]
        J = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=2)
        det = np.linalg.det(J)
        self.vol = np.abs(det) / 6.0
        Ji = np.linalg.inv(J)                                   # rows = grad lambda_1..3
        self.G = np.concatenate([-Ji.sum(axis=1, keepdims=True), Ji], axis=1)
        n = len(self.P)
        self.w = np.bincount(self.C.ravel(), np.repeat(self.vol / 4.0, 4), n)
        F = np.concatenate([self.C[:, [0, 1, 2]], self.C[:, [0, 1, 3]],
                            self.C[:, [0, 2, 3]], self.C[:, [1, 2, 3]]])
        Fs = np.sort(F, axis=1)
        u, cnt = np.unique(Fs, axis=0, return_counts=True)
        self.ext = u[cnt == 1]
        A = 0.5 * np.linalg.norm(np.cross(self.P[self.ext[:, 1]] - self.P[self.ext[:, 0]],
                                          self.P[self.ext[:, 2]] - self.P[self.ext[:, 0]]), axis=1)
        self.a = np.bincount(self.ext.ravel(), np.repeat(A / 3.0, 3), n)

    @property
    def n(self):
        return len(self.P)

    @property
    def V(self):
        return self.w.sum()

    def stiffness(self, D: np.ndarray) -> sp.csr_matrix:
        """D: (m,) scalar or (m,3,3) tensor per cell, mm^2/h."""
        if D.ndim == 1:
            KT = np.einsum("tia,tja->tij", self.G, self.G) * D[:, None, None]
        else:
            KT = np.einsum("tia,tab,tjb->tij", self.G, D, self.G)
        KT *= self.vol[:, None, None]
        I = np.repeat(self.C, 4, axis=1).ravel()
        Jx = np.tile(self.C, (1, 4)).ravel()
        K = sp.coo_matrix((KT.ravel(), (I, Jx)), shape=(self.n, self.n)).tocsr()
        K.sum_duplicates()
        return K

    def tissue_weights(self) -> np.ndarray:
        """(3,n) W_ji = sum over tets of label j containing node i of |T|/4."""
        W = np.zeros((3, self.n))
        for j in (1, 2, 3):
            m = self.label == j
            W[j - 1] = np.bincount(self.C[m].ravel(), np.repeat(self.vol[m] / 4.0, 4), self.n)
        return W


def load_base_mesh() -> Mesh:
    f = np.load(CORE / "04_DATA/geometry/fem_input.npz")
    rf = np.load(CORE / "04_DATA/plotting_assets/representative_fields.npz")
    # cell-wise coefficients and labels are stored in the cell order of representative_fields.npz
    assert np.array_equal(f["points"], rf["points_mm"])
    return Mesh(rf["points_mm"].astype(float), rf["cells"].astype(np.int64), rf["tissue_cell"].astype(np.int64))


def base_fields() -> dict:
    rf = np.load(CORE / "04_DATA/plotting_assets/representative_fields.npz")
    return {"LiteratureIsotropic": rf["canonical_D_mm2_h"].astype(float),
            "DataIsotropic": rf["isotropic_D_mm2_h"].astype(float),
            "DataTensor": rf["tensor_D_mm2_h"].astype(float)}


def checkpoint_boundary_data() -> np.ndarray:
    """(5,n) boundary concentration at the five public checkpoints."""
    return np.load(CORE / "04_DATA/geometry/fem_input.npz")["boundary"].astype(float)


# ----------------------------------------------------------------------------- boundary curve
def fit_boundary_curve(Bdata: np.ndarray, mask: np.ndarray, iters: int = 200, times_s=CHECKPOINT_S):
    """Vectorised Levenberg-Marquardt fit of B = a*tau*exp(-b*tau) per boundary vertex.

    Mirrors the miniapp: start (a,b)=(0.2,1.0); parameters held in [0,10]x[0,5]."""
    tau = np.asarray(times_s, float) / 86400.0
    y = Bdata[:, mask].T                                       # (nb,5)
    nb = y.shape[0]
    p = np.tile([0.2, 1.0], (nb, 1))
    lam = np.full(nb, 1e-3)

    def resid(p):
        e = np.exp(-p[:, 1:2] * tau[None, :])
        return p[:, 0:1] * tau[None, :] * e - y, e

    r, e = resid(p)
    cost = (r ** 2).sum(1)
    for _ in range(iters):
        Ja = tau[None, :] * e
        Jb = -p[:, 0:1] * tau[None, :] ** 2 * e
        A11 = (Ja * Ja).sum(1); A12 = (Ja * Jb).sum(1); A22 = (Jb * Jb).sum(1)
        g1 = (Ja * r).sum(1); g2 = (Jb * r).sum(1)
        M11 = A11 * (1 + lam); M22 = A22 * (1 + lam)
        det = M11 * M22 - A12 ** 2
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        d1 = -(M22 * g1 - A12 * g2) / det
        d2 = -(-A12 * g1 + M11 * g2) / det
        pn = p + np.stack([d1, d2], 1)
        inside = (pn[:, 0] >= 0) & (pn[:, 0] <= 10) & (pn[:, 1] >= 0) & (pn[:, 1] <= 5)
        rn, en = resid(pn)
        cn = (rn ** 2).sum(1)
        ok = (cn < cost) & inside                 # out-of-bounds steps are rejected, as the miniapp's penalty does
        p[ok] = pn[ok]; r[ok] = rn[ok]; e[ok] = en[ok]; cost[ok] = cn[ok]
        lam = np.where(ok, lam / 3.0, lam * 4.0)
        lam = np.clip(lam, 1e-12, 1e12)
    params = np.zeros((Bdata.shape[1], 2))
    params[mask] = p
    return params


def recover_official_boundary(mesh: "Mesh", D: np.ndarray, ref_id: str = "L22", sD: float = 1.0, r: float = 0.018):
    """Recover the per-vertex (a,b) the official solver actually used.

    The miniapp's own Levenberg-Marquardt stops at slightly different (a,b) than a fully
    converged fit at a minority of vertices. Because the discrete equations are linear, the
    boundary input of an archived official run is recovered exactly from its saved states,
    and (a,b) are then fitted to that input over all 74 positive saved times."""
    t = official_clock()
    c = np.load(EXT / f"04_DATA/trajectories/{ref_id}.npz")["NATIVE"].astype(float) / np.sqrt(mesh.w / mesh.V)
    K = mesh.stiffness(D)
    bm = mesh.a > 0
    Bimp = np.zeros((len(t), mesh.n))
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        R = mesh.w * (c[i] - c[i - 1]) / dt + sD * (K @ c[i]) + mesh.w * r * c[i]
        Bimp[i, bm] = PHI_SAS / (mesh.a[bm] * K_SURF) * (R[bm] + mesh.a[bm] * K_SURF * c[i][bm] / PHI_BRAIN)
    tau = t[1:, None] / 24.0
    # log-linear least squares: log(B/tau) = log a - b tau (exact when B has the official form),
    # then Gauss-Newton refinement in the original variables
    Y = Bimp[1:, bm]
    pos = Y > 1e-12
    L = np.where(pos, np.log(np.where(pos, Y, 1.0) / tau), 0.0)
    wts = pos * Y                                             # weight by magnitude
    S0 = wts.sum(0); S1 = (wts * tau).sum(0); S2 = (wts * tau ** 2).sum(0)
    T0 = (wts * L).sum(0); T1 = (wts * tau * L).sum(0)
    det = S0 * S2 - S1 ** 2
    det = np.where(np.abs(det) < 1e-300, 1e-300, det)
    loga = (S2 * T0 - S1 * T1) / det
    b = -(S0 * T1 - S1 * T0) / det
    a = np.exp(loga)
    for _ in range(30):
        e = np.exp(-b[None] * tau)
        r = a[None] * tau * e - Y
        Ja = tau * e; Jb = -a[None] * tau ** 2 * e
        A11 = (Ja * Ja).sum(0); A12 = (Ja * Jb).sum(0); A22 = (Jb * Jb).sum(0)
        g1 = (Ja * r).sum(0); g2 = (Jb * r).sum(0)
        dd = A11 * A22 - A12 ** 2
        dd = np.where(np.abs(dd) < 1e-300, 1e-300, dd)
        a = a - (A22 * g1 - A12 * g2) / dd
        b = b - (-A12 * g1 + A11 * g2) / dd
    params = np.zeros((mesh.n, 2))
    params[bm, 0], params[bm, 1] = a, b
    fitted = params[bm, 0][None] * tau * np.exp(-params[bm, 1][None] * tau)
    misfit = float(np.linalg.norm(fitted - Bimp[1:, bm]) / np.linalg.norm(Bimp[1:, bm]))
    return params, misfit


class Boundary:
    """Boundary concentration B(t) at every node (zero away from the surface)."""

    def __init__(self, params: np.ndarray, scale: float = 1.0, time_scale: float = 1.0,
                 region: np.ndarray | None = None):
        self.params = params
        self.scale = scale
        self.time_scale = time_scale          # >1 = faster pulse (time compressed)
        self.region = region                  # optional 0/1 mask for partial-surface inflow

    def __call__(self, t_h: float) -> np.ndarray:
        tau = self.time_scale * t_h / 24.0
        B = self.scale * self.params[:, 0] * tau * np.exp(-self.params[:, 1] * tau)
        if self.region is not None:
            B = B * self.region
        return B


# ----------------------------------------------------------------------------- forward solver
@dataclass
class Physics:
    k: float = K_SURF
    phi_brain: float = PHI_BRAIN
    phi_sas: float = PHI_SAS


def official_clock() -> np.ndarray:
    """The 75 saved times (h) of the official runs; a copy in results/ is used when the data extension is absent."""
    f = EXT / "04_DATA/trajectories/L22.npz"
    if f.exists():
        return np.load(f)["times_hours"].astype(float)
    return np.load(Path(__file__).resolve().parents[1] / "results/official_clock_hours.npy")


class Solver:
    """Backward-Euler solver with factorisations cached per (s_D, r, dt)."""

    def __init__(self, mesh: Mesh, K: sp.csr_matrix, bc: Boundary, times_h: np.ndarray,
                 phys: Physics = Physics(), method: str = "cg"):
        self.mesh, self.K, self.bc, self.t = mesh, K, bc, np.asarray(times_h, float)
        self.phys, self.method = phys, method
        self._cache = {}
        self.Bt = np.stack([bc(t) for t in self.t])            # (nt,n)

    def _factor(self, sD, r, dt):
        key = (round(sD, 14), round(r, 14), round(dt, 12))
        if key not in self._cache:
            m, ph = self.mesh, self.phys
            diag = m.w * (1.0 + r * dt) + dt * m.a * ph.k / ph.phi_brain
            A = (sp.diags(diag) + (dt * sD) * self.K).tocsc()
            if self.method == "direct":
                lu = sla.splu(A)
                self._cache[key] = lambda b, x0=None: lu.solve(b)
            elif self.method == "amg":
                import pyamg
                ml = pyamg.smoothed_aggregation_solver(A.tocsr(), symmetry="symmetric", max_coarse=500)

                def solve(b, x0=None, ml=ml):
                    return ml.solve(b, x0=x0, tol=1e-11, accel="cg", maxiter=200)
                self._cache[key] = solve
            else:
                Dinv = 1.0 / A.diagonal()
                Mpre = sla.LinearOperator(A.shape, matvec=lambda x: Dinv * x)

                def solve(b, x0=None, A=A, Mpre=Mpre):
                    try:
                        x, info = sla.cg(A, b, x0=x0, M=Mpre, rtol=1e-11, atol=0.0, maxiter=5000)
                    except TypeError:                     # SciPy < 1.12
                        x, info = sla.cg(A, b, x0=x0, M=Mpre, tol=1e-11, atol=0.0, maxiter=5000)
                    if info != 0:
                        raise RuntimeError(f"CG did not converge ({info})")
                    return x
                self._cache[key] = solve
            if len(self._cache) > 12:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def run(self, sD: float, r: float, sens: bool = False):
        """Return c (nt,n); with sens=True also dc/ds_D and dc/dr (nt,n) each."""
        m, ph = self.mesh, self.phys
        nt = len(self.t)
        c = np.zeros((nt, m.n))
        cs = np.zeros((nt, m.n)) if sens else None
        cr = np.zeros((nt, m.n)) if sens else None
        for i in range(1, nt):
            dt = self.t[i] - self.t[i - 1]
            solve = self._factor(sD, r, dt)
            rhs = m.w * c[i - 1] + dt * m.a * ph.k * self.Bt[i] / ph.phi_sas
            c[i] = solve(rhs, c[i - 1])
            if sens:
                cs[i] = solve(m.w * cs[i - 1] - dt * (self.K @ c[i]), cs[i - 1])
                cr[i] = solve(m.w * cr[i - 1] - dt * m.w * c[i], cr[i - 1])
        return (c, cs, cr) if sens else c

    def weighted(self, c):
        return c * np.sqrt(self.mesh.w / self.mesh.V)[None, :]


# ----------------------------------------------------------------------------- observations
class TissueROI:
    """Q_ji = W_ji / sqrt(W_j w_i) in weighted coordinates z_i = sqrt(w_i/V) c_i."""

    def __init__(self, mesh: Mesh, W: np.ndarray | None = None):
        W = mesh.tissue_weights() if W is None else W
        Wj = W.sum(axis=1)
        self.Q = sp.csr_matrix(W / np.sqrt(Wj[:, None] * mesh.w[None, :]))
        self.mean_op = sp.csr_matrix(W / Wj[:, None])          # c -> regional mean concentration
        self.Wj = Wj

    def __call__(self, z):                                   # z (nt,n) -> (nt,3)
        return (self.Q @ z.T).T

    def projector_parts(self, e):
        """Split residual e (nt,n) into retained/hidden squared energies (Gram-corrected)."""
        Q = self.Q
        G = (Q @ Q.T).toarray()
        Ginv = np.linalg.inv(G)
        Qe = (Q @ e.T)                                        # (3,nt)
        retained = float(np.sum(Qe * (Ginv @ Qe)))
        total = float(np.sum(e ** 2))
        return retained, total - retained


class Blocks:
    """Non-overlapping cubic blocks of width h (mm) on a fixed origin; Q rows sqrt(w_i/W_j)."""

    def __init__(self, mesh: Mesh, h: float, origin: np.ndarray):
        idx = np.floor((mesh.P - origin[None, :]) / h).astype(np.int64)
        _, lab = np.unique(idx, axis=0, return_inverse=True)
        lab = lab.ravel()
        nb = lab.max() + 1
        Wj = np.bincount(lab, mesh.w, nb)
        rows = lab
        self.Q = sp.csr_matrix((np.sqrt(mesh.w / Wj[lab]), (rows, np.arange(mesh.n))), shape=(nb, mesh.n))

    def __call__(self, z):
        return (self.Q @ z.T).T


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ----------------------------------------------------------------------------- fitting
def fit_two_scalar(solver: Solver, obs, y_truth, x0=(1.0, 0.018), lb=(0.6, 0.0), ub=(1.4, 0.036),
                   tmask=None, fix_r=None):
    """Least-squares fit of (s_D, r) to observations y_truth = obs(z_truth) (nt,m).

    Objective = sum_t ||obs(z_model)-y||^2 / sum_t ||y||^2 over tmask (default: t>0),
    the same normalisation as the archived fits. Exact forward sensitivities are used."""
    from scipy.optimize import least_squares
    t = solver.t
    tmask = (t > 0) if tmask is None else tmask
    y = y_truth[tmask]
    norm = np.linalg.norm(y)
    spans = np.array([ub[0] - lb[0], max(ub[1] - lb[1], 1e-12)])
    cache = {}

    def evaluate(u):
        key = tuple(np.round(u, 15))
        if key not in cache:
            p = lb + u * spans
            sD = p[0]
            r = p[1] if fix_r is None else fix_r
            c, cs, cr = solver.run(sD, r, sens=True)
            z, zs, zr = solver.weighted(c), solver.weighted(cs), solver.weighted(cr)
            res = (obs(z)[tmask] - y).ravel() / norm
            Js = obs(zs)[tmask].ravel() / norm * spans[0]
            Jr = obs(zr)[tmask].ravel() / norm * spans[1]
            J = np.stack([Js, Jr], 1) if fix_r is None else Js[:, None]
            cache.clear()
            cache[key] = (res, J, c)
        return cache[key]

    lbn, ubn = np.zeros(2), np.ones(2)
    u0 = (np.asarray(x0, float) - lb) / spans
    if fix_r is not None:
        lbn, ubn, u0 = lbn[:1], ubn[:1], u0[:1]
    sol = least_squares(lambda u: evaluate(u)[0], np.clip(u0, lbn, ubn), jac=lambda u: evaluate(u)[1],
                        bounds=(lbn, ubn), method="trf", xtol=1e-12, ftol=1e-14, gtol=1e-12, max_nfev=60)
    p = lb[: len(sol.x)] + sol.x * spans[: len(sol.x)]
    sD = float(p[0])
    r = float(p[1]) if fix_r is None else float(fix_r)
    res, J, c = evaluate(sol.x)
    return {"s_D": sD, "r": r, "observed_L2": float(np.linalg.norm(res)), "c": c,
            "nfev": int(sol.nfev), "status": int(sol.status), "active_bound": bool(np.any(sol.active_mask != 0))}


def timer():
    t0 = time.time()
    return lambda: time.time() - t0


# ----------------------------------------------------------------------------- refinement & variants
def refine_uniform(mesh: Mesh):
    """Red refinement: every tetrahedron -> 8 children (Bey), shortest interior diagonal.

    Returns (fine mesh, parent index per child cell, P1 interpolation matrix coarse->fine nodes).
    Geometry is unchanged (planar faces split exactly), labels are inherited."""
    C = mesh.C
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    E = np.sort(np.stack([C[:, i] for i in pairs.ravel()], 1).reshape(-1, 6, 2), axis=2)
    uniq, inv = np.unique(E.reshape(-1, 2), axis=0, return_inverse=True)
    inv = inv.reshape(-1, 6)
    n0 = mesh.n
    mid = n0 + inv                                              # (m,6) midpoint node ids
    P = np.vstack([mesh.P, 0.5 * (mesh.P[uniq[:, 0]] + mesh.P[uniq[:, 1]])])
    v0, v1, v2, v3 = C.T
    m01, m02, m03, m12, m13, m23 = mid.T
    corners = [np.stack(x, 1) for x in ((v0, m01, m02, m03), (v1, m01, m12, m13),
                                        (v2, m02, m12, m23), (v3, m03, m13, m23))]
    # octahedron: opposite pairs (m01,m23), (m02,m13), (m03,m12); split along the shortest
    opp = [(m01, m23, m02, m03, m13, m12), (m02, m13, m01, m03, m23, m12), (m03, m12, m01, m02, m23, m13)]
    L = np.stack([np.linalg.norm(P[a] - P[b], axis=1) for a, b, *_ in opp], 1)
    choice = np.argmin(L, 1)
    inner = []
    for k in range(4):
        tets = np.zeros((len(C), 4), np.int64)
        for d, (a, b, r0, r1, r2, r3) in enumerate(opp):
            ring = [r0, r1, r2, r3]
            sel = choice == d
            tets[sel] = np.stack([a[sel], b[sel], ring[k][sel], ring[(k + 1) % 4][sel]], 1)
        inner.append(tets)
    children = np.concatenate([np.stack(corners + inner, 1)], 0).reshape(-1, 4)  # (m*8,4), parent-major
    parent = np.repeat(np.arange(len(C)), 8)
    fine = Mesh(P, children, mesh.label[parent])
    rows = np.concatenate([np.arange(n0), n0 + np.arange(len(uniq)), n0 + np.arange(len(uniq))])
    cols = np.concatenate([np.arange(n0), uniq[:, 0], uniq[:, 1]])
    vals = np.concatenate([np.ones(n0), 0.5 * np.ones(len(uniq)), 0.5 * np.ones(len(uniq))])
    interp = sp.csr_matrix((vals, (rows, cols)), shape=(len(P), n0))
    return fine, parent, interp


class InterpolatedBoundary:
    """B on a refined mesh = P1 interpolant of the coarse boundary input (same continuous data)."""

    def __init__(self, coarse: Boundary, interp: sp.csr_matrix):
        self.coarse, self.interp = coarse, interp

    def __call__(self, t_h):
        return self.interp @ self.coarse(t_h)


def mmatrix_variant(K: sp.csr_matrix) -> sp.csr_matrix:
    """Drop wrong-sign (positive) off-diagonal couplings, keeping row sums (conservation).

    Equivalent to adding the minimal artificial diffusion that makes K an M-matrix, so
    backward Euler with lumped mass cannot create negative values."""
    Kc = K.tocoo()
    bad = (Kc.row != Kc.col) & (Kc.data > 0)
    Kp = sp.coo_matrix((Kc.data[bad], (Kc.row[bad], Kc.col[bad])), shape=K.shape).tocsr()
    return (K - Kp + sp.diags(np.asarray(Kp.sum(axis=1)).ravel())).tocsr()


def negative_mass_ratio(mesh: Mesh, c: np.ndarray) -> float:
    neg = (mesh.w[None] * np.maximum(-c, 0)).sum(1)
    pos = (mesh.w[None] * np.maximum(c, 0)).sum(1)
    return float(np.max(neg / np.maximum(pos, 1e-300)))


# ----------------------------------------------------------------------------- coefficient fields
def read_vtu_cell_array(path, name):
    """Read one zlib-compressed binary DataArray from a VTK XML file (no VTK dependency)."""
    import base64
    import re
    import zlib
    s = open(path).read()
    m = re.search(r'<DataArray[^>]*Name="%s"[^>]*>\s*([^<]+)</DataArray>' % name, s)
    txt = m.group(1).strip()
    nb = int(np.frombuffer(base64.b64decode(txt[:8])[:4], np.uint32)[0])
    hlen = ((4 * (3 + nb) + 2) // 3) * 4
    sizes = np.frombuffer(base64.b64decode(txt[:hlen]), np.uint32)[3:3 + nb]
    raw = base64.b64decode(txt[hlen:])
    out, o = [], 0
    for sz in sizes:
        out.append(zlib.decompress(raw[o:o + int(sz)]))
        o += int(sz)
    return np.frombuffer(b"".join(out), np.float64)


def atlas_multiplier(name: str, mesh: Mesh) -> np.ndarray:
    """E3/E4 cell multipliers in the cell order of the base mesh (matched by cell vertex sets)."""
    path = CORE / f"04_DATA/diffusion_fields/{name}_coefficient_mesh.vtu"
    mult = read_vtu_cell_array(path, "phase8h_scale")
    conn = read_vtu_cell_array_int(path, "connectivity").reshape(-1, 4)
    return _reorder_cells(conn, mesh.C, mult)


def read_vtu_cell_array_int(path, name):
    import base64
    import re
    import zlib
    s = open(path).read()
    m = re.search(r'<DataArray[^>]*Name="%s"[^>]*>\s*([^<]+)</DataArray>' % name, s)
    txt = m.group(1).strip()
    nb = int(np.frombuffer(base64.b64decode(txt[:8])[:4], np.uint32)[0])
    hlen = ((4 * (3 + nb) + 2) // 3) * 4
    sizes = np.frombuffer(base64.b64decode(txt[:hlen]), np.uint32)[3:3 + nb]
    raw = base64.b64decode(txt[hlen:])
    out, o = [], 0
    for sz in sizes:
        out.append(zlib.decompress(raw[o:o + int(sz)]))
        o += int(sz)
    return np.frombuffer(b"".join(out), np.int64)


def _reorder_cells(conn_src, conn_dst, values):
    a = np.sort(conn_src, 1)
    b = np.sort(conn_dst, 1)
    key = lambda x: x[:, 0] * 1_000_000_000_000 + x[:, 1] * 1_000_000 + x[:, 2] + x[:, 3] * 0  # noqa: E731
    ka = np.array([hash(tuple(r)) for r in a])
    kb = np.array([hash(tuple(r)) for r in b])
    order = {k: i for i, k in enumerate(ka)}
    idx = np.array([order[k] for k in kb])
    assert np.array_equal(a[idx], b)
    return values[idx]


def e2_multiplier(mesh: Mesh) -> np.ndarray:
    """E2: grey-matter diffusion x1.25 (cortical and subcortical grey), white matter x1."""
    return np.where(mesh.label == 2, 1.0, 1.25)


def correlated_multiplier(mesh: Mesh, ell: float, rms: float = 0.15, n_modes: int = 64, seed: int = 20261016):
    """E4-type positive multiplier with Gaussian-spectrum correlation length ell (mm).

    s(x) = sqrt(2/N) sum cos(k.x + phase), k ~ N(0, I/ell^2); m = 1 + rms * standardise(tanh(s)),
    standardised with cell-volume weights (mean 1, RMS deviation rms)."""
    rng = np.random.default_rng(seed)
    kvec = rng.normal(0.0, 1.0 / ell, size=(n_modes, 3))
    phase = rng.uniform(0, 2 * np.pi, n_modes)
    xc = mesh.P[mesh.C].mean(1)
    s = np.sqrt(2.0 / n_modes) * np.cos(xc @ kvec.T + phase[None]).sum(1)
    g = np.tanh(s)
    g = g - np.average(g, weights=mesh.vol)
    g = g / np.sqrt(np.average(g ** 2, weights=mesh.vol))
    return 1.0 + rms * g


def orientation_only_tensor(D_lit: np.ndarray, D_tensor: np.ndarray) -> np.ndarray:
    """Keep the canonical magnitude (trace/3 = D_lit) but the data tensor's directions and anisotropy."""
    tr3 = np.trace(D_tensor, axis1=1, axis2=2) / 3.0
    shape = D_tensor / tr3[:, None, None]
    return shape * D_lit[:, None, None]
