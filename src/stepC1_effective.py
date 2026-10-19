"""Step C1: compare archived fitted s_D with predictions made from the true field, without fitting.

Predictions of s_D = 1.1 x (effective magnitude of the true field) / (effective magnitude of the
model field), using
  * volume averages: arithmetic, geometric, harmonic of trace/3 ratio;
  * the 3-D small-contrast effective-medium formula <m> - var(m)/(3<m>) (multiplier fields);
  * numerical homogenisation: affine-Dirichlet cell problems on cubic blocks of width H,
    giving an effective tensor per block; blocks combined with volume weights or with the
    time-integrated gradient-energy weights of the canonical T2 run (where the tracer is).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
mesh = twin.load_base_mesh()
F = twin.base_fields()
D_lit = F["LiteratureIsotropic"]
iso = lambda d: d[:, None, None] * np.eye(3)[None]          # noqa: E731
cases = {
    "E2": iso(D_lit * twin.e2_multiplier(mesh)),
    "E3": iso(D_lit * twin.atlas_multiplier("E3", mesh)),
    "E4": iso(D_lit * twin.atlas_multiplier("E4", mesh)),
    "E5_DataIsotropic": iso(F["DataIsotropic"]),
    "E6_DataTensor": F["DataTensor"],
}
multipliers = {"E2": twin.e2_multiplier(mesh), "E3": twin.atlas_multiplier("E3", mesh),
               "E4": twin.atlas_multiplier("E4", mesh)}

# gradient-energy weights per cell from the canonical T2 truth (official run)
T2 = np.load(twin.EXT / "04_DATA/trajectories/T2.npz")
c = T2["NATIVE"].astype(float) / np.sqrt(mesh.w / mesh.V)
t = T2["times_hours"]
Ew = np.zeros(len(mesh.C))
for i in range(1, len(t)):
    g = np.einsum("tia,ti->ta", mesh.G, c[i][mesh.C])
    Ew += 1.1 * D_lit * (g ** 2).sum(1) * (t[i] - t[i - 1])
Ew *= mesh.vol


def block_homogenise(D, H):
    """Affine-Dirichlet upscaling on blocks of width H. Returns (block id per cell, Deff trace/3 per block)."""
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
        KII = K[I][:, I].tocsc()
        KIB = K[I][:, bnd]
        lu = sla.splu(KII)
        tr = 0.0
        Vb = mesh.vol[cells].sum()
        for e in range(3):
            u = mesh.P[nodes, e].copy()
            u[I] = lu.solve(-(KIB @ u[bnd]))
            grad = np.einsum("tia,ti->ta", mesh.G[cells], u[loc])
            flux = np.einsum("tab,tb->ta", D[cells], grad)
            tr += (mesh.vol[cells] * flux[:, e]).sum() / Vb
        deff[b] = tr / 3.0
    return blk, deff


rows = []
D_lit_t = iso(D_lit)
homog = {}
for H in (4.0, 8.0, 16.0):
    homog[("lit", H)] = block_homogenise(D_lit_t, H)
for name, D in cases.items():
    tr_true = np.trace(D, axis1=1, axis2=2) / 3.0
    ratio = tr_true / D_lit
    row = {"case": name,
           "arithmetic": 1.1 * np.average(tr_true, weights=mesh.vol) / np.average(D_lit, weights=mesh.vol),
           "geometric": 1.1 * np.exp(np.average(np.log(ratio), weights=mesh.vol)),
           "harmonic": 1.1 / np.average(1.0 / ratio, weights=mesh.vol),
           "energy_weighted": 1.1 * np.sum(Ew * ratio) / np.sum(Ew)}
    if name in multipliers:
        mu = multipliers[name]
        mean = np.average(mu, weights=mesh.vol)
        var = np.average((mu - mean) ** 2, weights=mesh.vol)
        row["effective_medium_3D"] = 1.1 * (mean - var / (3 * mean))
    for H in (4.0, 8.0, 16.0):
        blk, d_true = block_homogenise(D, H)
        _, d_lit = homog[("lit", H)]
        ok = np.isfinite(d_true) & np.isfinite(d_lit)
        vb = np.bincount(blk, mesh.vol, len(d_true))
        eb = np.bincount(blk, Ew, len(d_true))
        row[f"homog{int(H)}_vol"] = 1.1 * np.sum(vb[ok] * d_true[ok]) / np.sum(vb[ok] * d_lit[ok])
        row[f"homog{int(H)}_energy"] = 1.1 * np.sum(eb[ok] * d_true[ok]) / np.sum(eb[ok] * d_lit[ok])
    rows.append(row)
    print(row, flush=True)

pred = pd.DataFrame(rows)
atlas = pd.read_csv(twin.CORE / "03_RESULTS/phase8h_operator_atlas.csv")
fitted = []
for name in cases:
    br = name.split("_")[0]
    for op in ("NATIVE", "TISSUE_ROI"):
        q = atlas[(atlas.regime == "T2") & (atlas.branch == br) & (atlas.operator == op)]
        fitted.append({"case": name, "operator": op, "s_D_fit": float(q.estimated_s_D.iloc[0]),
                       "r_fit": float(q.estimated_r_h_inv.iloc[0])})
fit = pd.DataFrame(fitted).pivot(index="case", columns="operator", values="s_D_fit")
fit.columns = [f"fit_{c}" for c in fit.columns]
res = pred.set_index("case").join(fit)
res.to_csv(OUT / "stepC1_effective.csv")
print(res.round(4).to_string())
