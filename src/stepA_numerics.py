"""Step A: does the headline contrast survive mesh refinement and a positivity-preserving variant?

Headline case: tensor T2 truth (s_D=1.1, r=0.0135 1/h), canonical isotropic two-scalar fit to the
three tissue curves over all positive saved times.

Pass rule (set before running): the native/regional contrast (native_L2 / roi_L2) changes by
less than 10% relative between levels and variants, and the masking label does not flip.
Usage: python stepA_numerics.py <level> [standard|mmatrix] [x0_sD x0_r]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402
from experiments import headline  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
level = int(sys.argv[1])
variant = sys.argv[2] if len(sys.argv) > 2 else "standard"
x0 = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (1.0, 0.018)

mesh = twin.load_base_mesh()
fields = twin.base_fields()
params = np.load(OUT / "boundary_params_recovered.npy")
bc = twin.Boundary(params)
D_lit, D_ten = fields["LiteratureIsotropic"], fields["DataTensor"]
for _ in range(level):
    mesh, parent, interp = twin.refine_uniform(mesh)
    D_lit, D_ten = D_lit[parent], D_ten[parent]
    bc = twin.InterpolatedBoundary(bc, interp)
times = twin.official_clock()

K_ten, K_lit = mesh.stiffness(D_ten), mesh.stiffness(D_lit)
if variant == "mmatrix":
    K_ten, K_lit = twin.mmatrix_variant(K_ten), twin.mmatrix_variant(K_lit)
Kc = K_lit.tocoo()
off = Kc.row != Kc.col
out, curves, z_truth, z_fit = headline(mesh, K_ten, K_lit, bc, times, x0=x0, method="cg")
out.update({"level": level, "variant": variant, "nodes": mesh.n, "tets": len(mesh.C),
            "wrong_sign_edge_fraction_lit": float((Kc.data[off] > 0).mean()),
            "contrast_native_over_roi": out["native_L2"] / out["roi_L2"],
            "masked": bool(out["roi_L2"] <= 0.01 and out["native_L2"] > 0.01 and out["hidden_fraction"] >= 0.8)})
print(json.dumps(out, indent=2))
tag = f"L{level}_{variant}"
json.dump(out, open(OUT / f"stepA_{tag}.json", "w"), indent=2)
np.savez_compressed(OUT / f"stepA_{tag}_curves.npz", **curves)
