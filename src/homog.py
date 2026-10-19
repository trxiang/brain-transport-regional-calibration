"""Block homogenisation and tracer-weighting shared by Steps C1 and C2."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import twin


def gradient_energy_weights(mesh, c, t, D_scalar, sD=1.1):
    """Time-integrated s_D D |grad c|^2 |T| per cell for a trajectory c (nt,n)."""
    Ew = np.zeros(len(mesh.C))
    for i in range(1, len(t)):
        g = np.einsum("tia,ti->ta", mesh.G, c[i][mesh.C])
        Ew += sD * D_scalar * (g ** 2).sum(1) * (t[i] - t[i - 1])
    return Ew * mesh.vol


def block_homogenise(mesh, D, H):
    """Affine-Dirichlet cell problems on cubic blocks of width H (mm).

    D: (m,3,3). Returns (block id per cell, effective trace/3 per block)."""
    xc = mesh.P[mesh.C].mean(1)
    idx = np.floor((xc - xc.min(0)) / H).astype(np.int64)
    _, blk = np.unique(idx, axis=0, return_inverse=True)
    blk = blk.ravel()
    KT = np.einsum("tia,tab,tjb->tij", mesh.G, D, mesh.G) * mesh.vol[:, None, None]
    order = np.argsort(blk, kind="stable")
    bounds = np.searchsorted(blk[order], np.arange(blk.max() + 2))
    deff = np.full(blk.max() + 1, np.nan)
    for b in range(blk.max() + 1):
        cells = order[bounds[b]:bounds[b + 1]]
        if len(cells) < 8:
            continue
        Cb = mesh.C[cells]
        nodes, loc = np.unique(Cb, return_inverse=True)
        loc = loc.reshape(-1, 4)
        nl = len(nodes)
        K = sp.coo_matrix((KT[cells].ravel(), (np.repeat(loc, 4, 1).ravel(), np.tile(loc, (1, 4)).ravel())),
                          shape=(nl, nl)).tocsr()
        Fc = np.sort(np.concatenate([loc[:, [0, 1, 2]], loc[:, [0, 1, 3]], loc[:, [0, 2, 3]], loc[:, [1, 2, 3]]]), 1)
        u_, cnt = np.unique(Fc, axis=0, return_counts=True)
        bnd = np.zeros(nl, bool)
        bnd[u_[cnt == 1].ravel()] = True
        I = ~bnd
        if I.sum() == 0:
            continue
        lu = sla.splu(K[I][:, I].tocsc())
        KIB = K[I][:, bnd]
        tr, Vb = 0.0, mesh.vol[cells].sum()
        for e in range(3):
            u = mesh.P[nodes, e].copy()
            u[I] = lu.solve(-(KIB @ u[bnd]))
            grad = np.einsum("tia,ti->ta", mesh.G[cells], u[loc])
            flux = np.einsum("tab,tb->ta", D[cells], grad)
            tr += (mesh.vol[cells] * flux[:, e]).sum() / Vb
        deff[b] = tr / 3.0
    return blk, deff


def predictions(mesh, D_true, D_lit, Ew, Hs=(4.0, 8.0, 16.0), lit_cache=None):
    """Predicted s_D (x1.1) from several effective-value rules."""
    iso = lambda d: d[:, None, None] * np.eye(3)[None]      # noqa: E731
    Dt = D_true if D_true.ndim == 3 else iso(D_true)
    tr_true = np.trace(Dt, axis1=1, axis2=2) / 3.0
    ratio = tr_true / D_lit
    out = {"arithmetic": 1.1 * np.average(tr_true, weights=mesh.vol) / np.average(D_lit, weights=mesh.vol),
           "harmonic": 1.1 / np.average(1.0 / ratio, weights=mesh.vol),
           "geometric": 1.1 * np.exp(np.average(np.log(ratio), weights=mesh.vol)),
           "energy_weighted": 1.1 * np.sum(Ew * ratio) / np.sum(Ew)}
    lit_cache = {} if lit_cache is None else lit_cache
    for H in Hs:
        if H not in lit_cache:
            lit_cache[H] = block_homogenise(mesh, iso(D_lit), H)
        blk, d_lit = lit_cache[H]
        _, d_true = block_homogenise(mesh, Dt, H)
        ok = np.isfinite(d_true) & np.isfinite(d_lit)
        vb = np.bincount(blk, mesh.vol, len(d_true))
        eb = np.bincount(blk, Ew, len(d_true))
        out[f"homog{int(H)}_vol"] = 1.1 * np.sum(vb[ok] * d_true[ok]) / np.sum(vb[ok] * d_lit[ok])
        out[f"homog{int(H)}_energy"] = 1.1 * np.sum(eb[ok] * d_true[ok]) / np.sum(eb[ok] * d_lit[ok])
    return out
