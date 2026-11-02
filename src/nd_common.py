"""Shared tools for the new-directions study (November 2026).

Q1  error budget: can modelling and measurement errors fake "enhanced diffusion"?
Q2  advection: can regional or voxel data tell tissue flow from enhanced diffusion?
Q4  the real Gonzo concentration maps.

Builds on twin.py: same mesh, discretisation and units (mm, h, mmol/L). The model is

    dc/dt + div(v c) = div(s_D D(x) grad c) - r c

with the Robin exchange of twin.py on the whole exterior surface. The velocity v acts on the
tissue concentration c (v = Darcy velocity / porosity). Where v points into the brain, the
entering fluid carries the subarachnoid fluid concentration B/phi_SAS (tissue concentration
phi_brain*B/phi_SAS); where it points out, the tissue concentration leaves with the fluid.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import twin

GONZO = Path(os.environ.get("GONZO_DATA", twin.REPO.parent / "EXTERNAL_DATA/gonzo"))
SCAN_H = twin.CHECKPOINT_S[1:] / 3600.0          # 4.366, 23.89, 47.86, 69.90 h
UM_PER_MIN = 0.06                                  # 1 um/min in mm/h
OUTER_TAGS, VENT_TAGS = (5, 6, 8), (11, 12, 13)    # Gonzo facet tags: pial/exterior and ventricle walls
# Geometry: "gonzo" (default) or another subject built by vinje_geometry.py into results/nd_<geom>/geometry.npz
GEOM = os.environ.get("ND_GEOM", "gonzo")
ND_OUT = twin.REPO / ("results/nd" if GEOM == "gonzo" else f"results/nd_{GEOM}")


def other_geometry() -> dict:
    return dict(np.load(ND_OUT / "geometry.npz"))


# ----------------------------------------------------------------------------- setup
def exterior_tags(mesh: twin.Mesh) -> np.ndarray:
    """Gonzo facet tag of every exterior triangle of mesh.ext (cached next to the Gonzo data)."""
    if GEOM != "gonzo":
        return other_geometry()["ext_tags"]
    cache = GONZO / "exterior_face_tags.npy"
    if cache.exists():
        tags = np.load(cache)
        if len(tags) == len(mesh.ext):
            return tags
    import h5py
    g = h5py.File(GONZO / "mri_processed_data/sub-01/modeling/resolution32/data.hdf", "r")
    v = g["domain/boundaries/values"][:]
    t = np.sort(g["domain/boundaries/topology"][:], 1)
    key = lambda a: a[:, 0] * 10 ** 10 + a[:, 1] * 10 ** 5 + a[:, 2]  # noqa: E731
    lookup = dict(zip(key(t), v))
    tags = np.array([lookup[k] for k in key(np.sort(mesh.ext, 1))])
    np.save(cache, tags)
    return tags


class Setup:
    """Mesh, coefficient fields, boundary input, clock, observation operators and surface geometry."""

    def __init__(self):
        if GEOM == "gonzo":
            self.mesh = m = twin.load_base_mesh()
            self.fields = twin.base_fields()
            self.params = np.load(twin.REPO / "results/boundary_params_recovered.npy")
            outer_tags, vent_tags = OUTER_TAGS, VENT_TAGS
        else:
            g = other_geometry()
            self.mesh = m = twin.Mesh(g["P"], g["C"].astype(np.int64), g["label"].astype(np.int64))
            self.fields = {k: g[k] for k in ("LiteratureIsotropic", "DataIsotropic", "DataTensor")}
            self.params = g["params"]
            outer_tags, vent_tags = tuple(g["outer_tags"]), tuple(g["vent_tags"])
        self.t = twin.official_clock()
        self.scan_idx = np.array([int(np.argmin(np.abs(self.t - s))) for s in SCAN_H])
        assert np.allclose(self.t[self.scan_idx], SCAN_H, atol=1e-6)
        self.roi = twin.TissueROI(m)
        tags = exterior_tags(m)
        self.bnodes = np.unique(m.ext.ravel())
        self.vent_nodes = np.unique(m.ext[np.isin(tags, vent_tags)].ravel())
        self.outer_nodes = np.setdiff1d(np.unique(m.ext[np.isin(tags, outer_tags)].ravel()), self.vent_nodes)
        d, j = cKDTree(m.P[self.bnodes]).query(m.P)
        self.dist = d                                  # distance to the nearest surface node, mm
        self.nearest_b = self.bnodes[j]

    def tmask(self, max_h: float | None = None, scans_only: bool = True) -> np.ndarray:
        mask = np.zeros(len(self.t), bool)
        if scans_only:
            mask[self.scan_idx] = True
        else:
            mask[1:] = True
        if max_h is not None:
            mask &= self.t <= max_h + 1e-9
        return mask


# ----------------------------------------------------------------------------- boundary inputs
class LinearBoundary:
    """B(t) piecewise linear in time through sampled values, held constant after the last sample.

    This is how Vinje et al. (2023) prescribe the subarachnoid concentration between MRI sessions."""

    def __init__(self, times_h: np.ndarray, samples: np.ndarray):
        self.th = np.asarray(times_h, float)
        self.S = np.asarray(samples, float)          # (len(times_h), n)

    def __call__(self, t_h: float) -> np.ndarray:
        if t_h >= self.th[-1]:
            return self.S[-1].copy()
        k = int(np.searchsorted(self.th, t_h, side="right") - 1)
        k = max(k, 0)
        a = (t_h - self.th[k]) / (self.th[k + 1] - self.th[k])
        return (1 - a) * self.S[k] + a * self.S[k + 1]


class ScaledBoundary:
    def __init__(self, bc, scale: float):
        self.bc, self.scale = bc, scale

    def __call__(self, t_h):
        return self.scale * self.bc(t_h)


def sampled_linear(bc, times_h=SCAN_H) -> LinearBoundary:
    """What an analyst sees: bc sampled at the scan times (zero at injection), linearly interpolated."""
    th = np.concatenate([[0.0], times_h])
    return LinearBoundary(th, np.stack([bc(t) for t in th]))


# ----------------------------------------------------------------------------- advection
def darcy_field(mesh: twin.Mesh, hi_nodes: np.ndarray, lo_nodes: np.ndarray, mean_speed_mm_h: float = UM_PER_MIN):
    """Divergence-free cellwise velocity v = -grad p, K p = 0, p = 1 on hi_nodes, 0 on lo_nodes,
    no flux elsewhere; scaled to the given volume-weighted mean speed."""
    K1 = mesh.stiffness(np.ones(len(mesh.C)))
    fixed = np.zeros(mesh.n, bool)
    fixed[hi_nodes] = fixed[lo_nodes] = True
    p = np.zeros(mesh.n)
    p[hi_nodes] = 1.0
    free = ~fixed
    A = K1[free][:, free].tocsc()
    b = -K1[free][:, fixed] @ p[fixed]
    p[free] = sla.spsolve(A, b)
    v = -np.einsum("tia,ti->ta", mesh.G, p[mesh.C])
    speed = np.average(np.linalg.norm(v, axis=1), weights=mesh.vol)
    return p, v * (mean_speed_mm_h / speed)


def flux_field(mesh: twin.Mesh, in_nodes: np.ndarray, out_nodes: np.ndarray | None = None,
               mean_speed_mm_h: float = UM_PER_MIN):
    """Velocity v = -grad p for prescribed fluxes: uniform inflow per unit area on in_nodes and either
    uniform outflow per unit area on out_nodes or, when out_nodes is None, uniform drainage per unit
    volume throughout the tissue (fluid leaving along veins). Scaled to the volume-weighted mean speed.

    Returns (p, v, drainage) with drainage the fluid drainage rate (1/h) when out_nodes is None, else 0.
    With drainage the discrete divergence of v is -drainage at interior nodes, and advection_operator
    turns it into a solute sink of the same rate (solute leaves with the draining fluid)."""
    K1 = mesh.stiffness(np.ones(len(mesh.C)))
    q = np.zeros(mesh.n)                               # target nodal outflow (negative = inflow)
    q[in_nodes] -= mesh.a[in_nodes] / mesh.a[in_nodes].sum()
    if out_nodes is None:
        q += mesh.w / mesh.V
    else:
        q[out_nodes] += mesh.a[out_nodes] / mesh.a[out_nodes].sum()
    pin = int(np.argmax(mesh.P[:, 2]))
    free = np.ones(mesh.n, bool)
    free[pin] = False
    p = np.zeros(mesh.n)
    p[free] = sla.spsolve(K1[free][:, free].tocsc(), -q[free])
    v = -np.einsum("tia,ti->ta", mesh.G, p[mesh.C])
    speed = np.average(np.linalg.norm(v, axis=1), weights=mesh.vol)
    scale = mean_speed_mm_h / speed
    drainage = scale / mesh.V if out_nodes is None else 0.0     # total inflow (scaled) / volume
    return p * scale, v * scale, drainage


def advection_operator(mesh: twin.Mesh, v: np.ndarray, stabilise: bool = False):
    """Assemble the conservative advection operator for a cellwise velocity v (mm/h).

    Returns (L, q_in): L is the left-hand-side operator (interior Galerkin coupling plus outflow
    on the diagonal) and q_in >= 0 the inflow rate per boundary node (mm^3/h), so that the
    semi-discrete equation reads  w dc/dt + L c = q_in * c_inflow + ...  With stabilise=True the
    interior coupling gets discrete upwinding (Kuzmin), which adds artificial diffusion v h / 2."""
    gv = np.einsum("tia,ta->ti", mesh.G, v) * (mesh.vol / 4.0)[:, None]          # (m,4): |T|/4 grad(phi_i).v
    I = np.repeat(mesh.C, 4, axis=1).ravel()
    J = np.tile(mesh.C, (1, 4)).ravel()
    vals = np.repeat(-gv, 4, axis=1).ravel()                                       # M_ij = -|T|/4 grad(phi_i).v
    M = sp.coo_matrix((vals, (I, J)), shape=(mesh.n, mesh.n)).tocsr()
    M.sum_duplicates()
    q = -np.asarray(M.sum(axis=1)).ravel()                                          # boundary flux, >0 outflow
    if stabilise:
        off = M - sp.diags(M.diagonal())
        Dm = off.maximum(off.T)
        Dm = Dm.multiply(Dm > 0).tocsr()
        M = (M - Dm + sp.diags(np.asarray(Dm.sum(axis=1)).ravel())).tocsr()
    L = (M + sp.diags(np.maximum(q, 0.0))).tocsr()
    return L, np.maximum(-q, 0.0), q


def cell_peclet(mesh: twin.Mesh, v: np.ndarray, D: np.ndarray, sD: float = 1.0) -> np.ndarray:
    """Cell Peclet number |v| h / (2 sD D) with h the longest edge (scalar D or tensor trace/3)."""
    X = mesh.P[mesh.C]
    h = np.max([np.linalg.norm(X[:, a] - X[:, b], axis=1) for a, b in
                ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], axis=0)
    Ds = D if D.ndim == 1 else np.trace(D, axis1=1, axis2=2) / 3.0
    return np.linalg.norm(v, axis=1) * h / (2.0 * sD * Ds)


# ----------------------------------------------------------------------------- solver
class GenSolver:
    """Backward Euler for  w dc/dt + s_D K c + vs L c + r w c + Robin = Robin source + vs q_in phi_b B/phi_SAS.

    vs is the velocity scale in multiples of the unit field behind L (um/min when L is built from
    a 1 um/min field). Symmetric systems (vs = 0) use CG as in twin.Solver; others use sparse LU."""

    def __init__(self, mesh, K, bc, times_h, phys=twin.Physics(), L=None, q_in=None, iterative=None):
        self.mesh, self.K, self.t, self.phys = mesh, K, np.asarray(times_h, float), phys
        self.L, self.q_in = L, q_in
        # flow systems by Jacobi-preconditioned BiCGStab instead of LU (default on the larger other geometries;
        # the Gonzo results used LU, and the two agree to 3e-8)
        self.iterative = (GEOM != "gonzo") if iterative is None else iterative
        self.Bt = np.stack([bc(t) for t in self.t])
        self._cache = {}

    def _factor(self, sD, r, vs, dt, kx=1.0):
        key = (round(sD, 14), round(r, 14), round(vs, 14), round(kx, 14), round(dt, 12))
        if key not in self._cache:
            m, ph = self.mesh, self.phys
            diag = m.w * (1.0 + r * dt) + dt * m.a * (kx * ph.k) / ph.phi_brain
            A = sp.diags(diag) + (dt * sD) * self.K
            if vs != 0.0 and self.iterative:
                A = (A + (dt * vs) * self.L).tocsr()
                Dinv = 1.0 / A.diagonal()
                Mpre = sla.LinearOperator(A.shape, matvec=lambda x, Dinv=Dinv: Dinv * x)

                def solve(b, x0=None, A=A, Mpre=Mpre):
                    try:
                        x, info = sla.bicgstab(A, b, x0=x0, M=Mpre, rtol=1e-11, atol=0.0, maxiter=20000)
                    except TypeError:                     # SciPy < 1.12
                        x, info = sla.bicgstab(A, b, x0=x0, M=Mpre, tol=1e-11, atol=0.0, maxiter=20000)
                    if info != 0:
                        raise RuntimeError(f"BiCGStab did not converge ({info})")
                    return x
                self._cache[key] = solve
            elif vs != 0.0:
                # LU factors are large (~1 GB): hold one at a time (the clock switches dt only 8 times per run)
                self._cache.clear()
                A = (A + (dt * vs) * self.L).tocsc()
                lu = sla.splu(A)
                self._cache[key] = lambda b, x0=None, lu=lu: lu.solve(b)
            else:
                A = A.tocsr()
                Dinv = 1.0 / A.diagonal()
                Mpre = sla.LinearOperator(A.shape, matvec=lambda x, Dinv=Dinv: Dinv * x)

                def solve(b, x0=None, A=A, Mpre=Mpre):
                    try:
                        x, info = sla.cg(A, b, x0=x0, M=Mpre, rtol=1e-11, atol=0.0, maxiter=10000)
                    except TypeError:                     # SciPy < 1.12
                        x, info = sla.cg(A, b, x0=x0, M=Mpre, tol=1e-11, atol=0.0, maxiter=10000)
                    if info != 0:
                        raise RuntimeError(f"CG did not converge ({info})")
                    return x
                self._cache[key] = solve
            if len(self._cache) > 14:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def run(self, sD: float, r: float, vs: float = 0.0, sens=(), kx: float = 1.0):
        """Return c (nt,n) and a dict of sensitivities for the names in sens ('sD', 'r', 'vs', 'kx').

        kx multiplies the exchange coefficient k of the Robin condition."""
        m, ph = self.mesh, self.phys
        nt = len(self.t)
        c = np.zeros((nt, m.n))
        S = {k: np.zeros((nt, m.n)) for k in sens}
        for i in range(1, nt):
            dt = self.t[i] - self.t[i - 1]
            solve = self._factor(sD, r, vs, dt, kx)
            fluid_in = ph.phi_brain * self.Bt[i] / ph.phi_sas
            rhs = m.w * c[i - 1] + dt * m.a * (kx * ph.k) * self.Bt[i] / ph.phi_sas
            if vs != 0.0:
                rhs = rhs + dt * vs * self.q_in * fluid_in
            c[i] = solve(rhs, c[i - 1])
            if "sD" in S:
                S["sD"][i] = solve(m.w * S["sD"][i - 1] - dt * (self.K @ c[i]), S["sD"][i - 1])
            if "r" in S:
                S["r"][i] = solve(m.w * S["r"][i - 1] - dt * m.w * c[i], S["r"][i - 1])
            if "vs" in S:
                S["vs"][i] = solve(m.w * S["vs"][i - 1] - dt * (self.L @ c[i]) + dt * self.q_in * fluid_in,
                                   S["vs"][i - 1])
            if "kx" in S:
                S["kx"][i] = solve(m.w * S["kx"][i - 1] + dt * m.a * ph.k * (self.Bt[i] / ph.phi_sas
                                                                         - c[i] / ph.phi_brain), S["kx"][i - 1])
        return c, S

    def weighted(self, c):
        return c * np.sqrt(self.mesh.w / self.mesh.V)[None, :]


# ----------------------------------------------------------------------------- observation models
def partial_volume(setup: Setup, c: np.ndarray, Bt: np.ndarray, f0: float = 0.5, ell: float = 1.0) -> np.ndarray:
    """Mix CSF signal into tissue values near the surface: c_obs = (1 - f) c + f B_nearest,
    f = f0 exp(-d / ell), d = distance to the nearest surface node (mm)."""
    f = f0 * np.exp(-setup.dist / ell)
    return (1.0 - f)[None, :] * c + f[None, :] * Bt[:, setup.nearest_b]


def add_noise(c: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    out = c + np.random.default_rng(seed).normal(0.0, sigma, c.shape)
    out[0] = 0.0
    return out


def roi_obs(setup: Setup):
    return setup.roi


def field_obs(z):
    return z


DEPTH_EDGES = (0.0, 1e-6, 2.0, 4.0, 6.0, np.inf)       # surface nodes, then 0-2, 2-4, 4-6, >6 mm


def depth_tissue_rows(setup: Setup, edges=DEPTH_EDGES) -> np.ndarray:
    """(nrow,n) tissue-volume weights split by distance to the nearest CSF surface: a depth profile per tissue."""
    W = setup.mesh.tissue_weights()
    rows = []
    for j in range(3):
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (setup.dist >= a) & (setup.dist < b)
            r = W[j] * sel
            if r.sum() > 0.002 * W[j].sum():
                rows.append(r)
    return np.array(rows)


def block_rows(setup: Setup, h: float = 8.0) -> sp.csr_matrix:
    """(nblock,n) node volumes w_i in each cubic block of width h mm."""
    bl = twin.Blocks(setup.mesh, h, setup.mesh.P.min(0) - 1e-6)
    return (bl.Q > 0).multiply(setup.mesh.w[None, :]).tocsr()


class WhitenedObs:
    """Region means c_bar_j = sum_i W_ji c_i / W_j, divided by their noise standard deviation under iid
    node noise sigma, acting on weighted coordinates z (as every obs in this module). For the native
    field (W = None) the observation is c_i / sigma at every node. Squared residual norms are chi^2."""

    def __init__(self, setup: Setup, W=None, sigma: float = 0.02):
        self.sw = np.sqrt(setup.mesh.w / setup.mesh.V)
        self.sigma = sigma
        if W is None:
            self.M = None
            self.nrow = setup.mesh.n
        else:
            W = sp.csr_matrix(W)
            Wj = np.asarray(W.sum(axis=1)).ravel()
            self.M = sp.diags(1.0 / Wj) @ W                        # c -> region means
            sd = sigma * np.sqrt(np.asarray(self.M.multiply(self.M).sum(axis=1)).ravel())
            self.M = (sp.diags(1.0 / sd) @ self.M).tocsr()
            self.nrow = W.shape[0]

    def __call__(self, z):
        c = z / self.sw[None, :]
        if self.M is None:
            return c / self.sigma
        return (self.M @ c.T).T


def whitened_operators(setup: Setup, sigma: float = 0.02) -> dict:
    """The four observation types of Q2: tissue curves, depth profiles, 8 mm blocks, native field."""
    return {"roi": WhitenedObs(setup, setup.mesh.tissue_weights(), sigma),
            "depth": WhitenedObs(setup, depth_tissue_rows(setup), sigma),
            "block8": WhitenedObs(setup, block_rows(setup, 8.0), sigma),
            "field": WhitenedObs(setup, None, sigma)}


# ----------------------------------------------------------------------------- fitting
DEFAULT_BOUNDS = {"sD": (0.1, 10.0), "r": (0.0, 0.6), "vs": (0.0, 20.0)}


def fit(solver: GenSolver, obs, y: np.ndarray, tmask: np.ndarray, names=("sD", "r"), starts=((1.0, 0.02),),
        fixed=None, bounds=None, max_nfev=80):
    """Least squares of obs(z_model) against y (both (nt,m) in weighted coordinates) over tmask.

    Objective = sum ||obs(z)-y||^2 / sum ||y||^2 over tmask; exact forward sensitivities; bounded
    trust-region reflective from every start; the best result is returned."""
    fixed = dict({"sD": 1.0, "r": 0.0, "vs": 0.0}, **(fixed or {}))
    bnd = dict(DEFAULT_BOUNDS, **(bounds or {}))
    lb = np.array([bnd[k][0] for k in names])
    ub = np.array([bnd[k][1] for k in names])
    spans = ub - lb
    yv = y[tmask]
    norm = np.linalg.norm(yv)
    cache = {}

    def evaluate(u):
        key = tuple(np.round(u, 15))
        if key not in cache:
            p = dict(fixed)
            p.update(zip(names, lb + u * spans))
            c, S = solver.run(p["sD"], p["r"], p["vs"], sens=names)
            res = (obs(solver.weighted(c))[tmask] - yv).ravel() / norm
            J = np.stack([obs(solver.weighted(S[k]))[tmask].ravel() / norm * spans[j]
                          for j, k in enumerate(names)], 1)
            cache.clear()
            cache[key] = (res, J, c)
        return cache[key]

    best = None
    for x0 in starts:
        u0 = np.clip((np.asarray(x0, float)[: len(names)] - lb) / spans, 0.0, 1.0)
        sol = least_squares(lambda u: evaluate(u)[0], u0, jac=lambda u: evaluate(u)[1], bounds=(0.0, 1.0),
                            method="trf", xtol=1e-12, ftol=1e-14, gtol=1e-12, max_nfev=max_nfev)
        res, J, c = evaluate(sol.x)
        cost = float(np.linalg.norm(res))
        if best is None or cost < best["observed_L2"]:
            p = dict(fixed)
            p.update(zip(names, lb + sol.x * spans))
            best = {**{k: float(v) for k, v in p.items()}, "observed_L2": cost, "c": c, "nfev": int(sol.nfev),
                    "status": int(sol.status), "active_bound": bool(np.any(sol.active_mask != 0)),
                    "start": list(map(float, x0))}
    return best


def score(setup: Setup, c_fit: np.ndarray, c_truth: np.ndarray, tmask: np.ndarray) -> dict:
    """Regional, native and hidden metrics of a fitted trajectory against a noiseless truth."""
    w = np.sqrt(setup.mesh.w / setup.mesh.V)[None, :]
    zf, zt = c_fit[tmask] * w, c_truth[tmask] * w
    roi = setup.roi
    retained, hidden = roi.projector_parts(zf - zt)
    return {"roi_L2": twin.rel_l2(roi(zf), roi(zt)), "native_L2": twin.rel_l2(zf, zt),
            "hidden_fraction": hidden / max(retained + hidden, 1e-300)}


def roi_means(setup: Setup, c: np.ndarray) -> np.ndarray:
    return (setup.roi.mean_op @ c.T).T


# ----------------------------------------------------------------------------- real data
def gonzo_data(setup: Setup) -> dict:
    """Real Gonzo concentrations and raw subarachnoid boundary values on the working-mesh nodes."""
    d = np.load(GONZO / "gonzo_nodal_data.npz")
    assert d["concentration_vtu"].shape[1] == setup.mesh.n
    return {"times_h": d["times_s"] / 3600.0, "c": d["concentration_vtu"], "B_raw": d["boundary_vtu"]}
