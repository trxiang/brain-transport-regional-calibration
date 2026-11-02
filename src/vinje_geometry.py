"""Build a second (and further) human brain geometry from the public Vinje et al. meshes.

Source: Zenodo 10.5281/zenodo.8036492 (CC BY 4.0), example meshes with DTI, FEniCS HDF5 format
(/mesh, /subdomains 1 grey 2 white 3 brainstem, /boundaries, DTI per cell). For mesh <id>:

  geometry   nodes, tetrahedra, tissue labels (grey, white, brainstem as the three regions)
  fields     DataTensor = DTI (mm^2/s) x 3600 x the miniapp water-to-gadobutrol ratio 3.5e-10/3e-9,
             eigenvalues clipped below at 10% of the mean diffusivity (non-SPD and near-zero cells);
             DataIsotropic = its trace / 3; LiteratureIsotropic = 0.504 (grey) / 0.396 mm^2/h
  CSF input  no tracer data come with these meshes: the Gonzo per-vertex curve parameters (a, b of
             B = a tau exp(-b tau)) are transferred to every exterior vertex from the nearest Gonzo
             exterior vertex of the same kind (outer surface or ventricle) after aligning the two
             brains by centroid and per-axis extent
  flow       the three Gonzo flow patterns rebuilt on this mesh (pial inflow with drainage, pial to
             ventricle flux, basal to dorsal Darcy), mean speed 1 um/min

Usage: python vinje_geometry.py <id> [h5 path]     writes results/nd_vinje<id>/geometry.npz and velocity_*.npy
"""
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402

RATIO = 3.5e-10 / 3e-9
SRC = Path(os.environ.get("VINJE_DATA", twin.REPO.parent / "EXTERNAL_DATA/vinje_meshes"))


def find_h5(mid):
    return SRC / "example_meshes" / mid / "parenchyma32_with_DTI.h5"


def read(h5):
    f = h5py.File(h5, "r")
    P = f["mesh/coordinates"][:].astype(float)
    C = f["mesh/topology"][:].astype(np.int64)
    assert np.array_equal(f["subdomains/topology"][:], f["mesh/topology"][:])
    label = f["subdomains/values"][:].astype(np.int64)
    bt, bv = f["boundaries/topology"][:].astype(np.int64), f["boundaries/values"][:].astype(np.int64)
    g = f["DTI"]                                     # DG0 tensor function: 9 dofs per cell
    x = g["x_cell_dofs"][:]
    assert np.all(np.diff(x) == 9)
    dti = np.zeros((len(C), 9))
    dti[g["cells"][:].astype(np.int64)] = g["vector_0"][:][g["cell_dofs"][:].reshape(-1, 9)]
    return P, C, label, bt, bv, dti, f


def clip_tensor(T):
    """Symmetrise and clip eigenvalues below at 10% of the volume-mean diffusivity."""
    T = 0.5 * (T + np.transpose(T, (0, 2, 1)))
    lam, V = np.linalg.eigh(T)
    floor = 0.1 * np.mean(np.clip(lam, 0, None).mean(1))
    lam_c = np.clip(lam, floor, None)
    return np.einsum("tij,tj,tkj->tik", V, lam_c, V), int(np.sum(lam.min(1) < floor))


def main(mid, h5=None):
    h5 = Path(h5) if h5 else find_h5(mid)
    P, C, label, bt, bv, dti, f = read(h5)
    out_dir = twin.REPO / f"results/nd_vinje{mid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = twin.Mesh(P, C, label)
    info = {"source": str(h5), "nodes": mesh.n, "cells": len(C), "volume_mm3": float(mesh.V),
            "labels": {int(k): int(v) for k, v in zip(*np.unique(label, return_counts=True))}}

    # diffusion fields (DTI per cell: 9 values)
    T = np.asarray(dti, float).reshape(len(C), 3, 3) * 3600.0 * RATIO
    Tc, nclip = clip_tensor(T)
    md = np.trace(Tc, axis1=1, axis2=2) / 3.0
    lit = np.where(label == 1, 0.504, 0.396)
    info.update(clipped_cells=nclip, tensor_trace3_mean=float(np.average(md, weights=mesh.vol)))

    # exterior tags: facet values matched to mesh.ext
    key = lambda a: a[:, 0] * 10 ** 12 + a[:, 1] * 10 ** 6 + a[:, 2]          # noqa: E731
    lookup = dict(zip(key(np.sort(bt, 1)), bv))
    ext_tags = np.array([lookup.get(k, -1) for k in key(np.sort(mesh.ext, 1))])
    tags, counts = np.unique(ext_tags, return_counts=True)
    # ventricle tags: facet groups lying away from the largest (pial) surface; Vinje meshes: 1 pial,
    # 2 lateral/third ventricles (against white matter), 3 aqueduct/fourth ventricle (against brainstem)
    area = {int(t): float(mesh.a[np.unique(mesh.ext[ext_tags == t])].sum()) for t in tags}
    outer = [max(area, key=area.get)]
    pial = cKDTree(P[np.unique(mesh.ext[ext_tags == outer[0]].ravel())])
    depth = {int(t): float(np.median(pial.query(P[np.unique(mesh.ext[ext_tags == t].ravel())])[0])) for t in tags}
    vent = [t for t in area if t not in outer and depth[t] > 1.0]
    outer += [t for t in area if t not in outer and t not in vent]
    info.update(ext_tags={int(t): int(c) for t, c in zip(tags, counts)}, tag_median_dist_to_pial=depth, tag_area=area,
                outer_tags=outer, vent_tags=vent)

    # CSF input transferred from Gonzo
    os.environ.setdefault("ND_GEOM", "gonzo")
    import nd_common as nd
    G = nd.Setup()
    gp = G.params
    gb_out, gb_vent = G.outer_nodes, G.vent_nodes
    lo_g, hi_g = G.mesh.P.min(0), G.mesh.P.max(0)
    lo_v, hi_v = P.min(0), P.max(0)
    Pa = lo_g + (P - lo_v) * (hi_g - lo_g) / (hi_v - lo_v)                  # Vinje nodes in Gonzo frame
    bnodes = np.unique(mesh.ext.ravel())
    vent_nodes = np.unique(mesh.ext[np.isin(ext_tags, vent)].ravel())
    outer_nodes = np.setdiff1d(bnodes, vent_nodes)
    params = np.zeros((mesh.n, 2))
    for mine, theirs in ((outer_nodes, gb_out), (vent_nodes, gb_vent)):
        if len(mine) and len(theirs):
            _, j = cKDTree(G.mesh.P[theirs]).query(Pa[mine])
            params[mine] = gp[theirs[j]]
    info["params_nonzero_boundary_frac"] = float(np.mean(np.abs(params[bnodes]).sum(1) > 0))
    np.savez_compressed(out_dir / "geometry.npz", P=P, C=C.astype(np.int32), label=label.astype(np.int8),
                        DataTensor=Tc, DataIsotropic=md, LiteratureIsotropic=lit, params=params,
                        ext_tags=ext_tags, outer_tags=np.array(outer), vent_tags=np.array(vent))

    # flow patterns on this mesh
    z = P[:, 2]
    zo = z[outer_nodes]
    _, v, _ = nd.flux_field(mesh, outer_nodes)
    np.save(out_dir / "velocity_pial_inflow_drainage.npy", v)
    if len(vent_nodes):
        _, v, _ = nd.flux_field(mesh, outer_nodes, vent_nodes)
        np.save(out_dir / "velocity_pial_to_ventricle_flux.npy", v)
    _, v = nd.darcy_field(mesh, outer_nodes[zo < np.quantile(zo, 0.2)], outer_nodes[zo > np.quantile(zo, 0.8)])
    np.save(out_dir / "velocity_basal_to_dorsal.npy", v)
    json.dump(info, open(out_dir / "geometry_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
