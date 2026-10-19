"""Shared experiment: generate a truth with one diffusion field, fit the canonical
two-scalar model to the three tissue curves, and report the paper's headline metrics."""
import numpy as np

import twin


def fit_and_score(mesh, K_model, bc, times, z_truth, phys=twin.Physics(), x0=(1.0, 0.018),
                  lb=(0.6, 0.0), ub=(1.4, 0.036), tmask=None, blocks_h=8.0, fix_r=None, method="cg"):
    """Fit (s_D, r) of the model operator K_model to the tissue curves of z_truth and score it."""
    roi = twin.TissueROI(mesh)
    solver = twin.Solver(mesh, K_model, bc, times, phys=phys, method=method)
    fit = twin.fit_two_scalar(solver, roi, roi(z_truth), x0=x0, lb=lb, ub=ub, tmask=tmask, fix_r=fix_r)
    z_fit = solver.weighted(fit["c"])
    pos = times > 0
    e = z_fit[pos] - z_truth[pos]
    retained, hidden = roi.projector_parts(e)
    blocks = twin.Blocks(mesh, blocks_h, mesh.P.min(0) - 1e-6)
    out = {
        "s_D_fit": fit["s_D"], "r_fit": fit["r"], "active_bound": fit["active_bound"], "nfev": fit["nfev"],
        "roi_L2": twin.rel_l2(roi(z_fit)[pos], roi(z_truth)[pos]),
        "native_L2": twin.rel_l2(z_fit[pos], z_truth[pos]),
        "hidden_fraction": hidden / (retained + hidden),
        f"block{int(blocks_h)}_L2": twin.rel_l2(blocks(z_fit)[pos], blocks(z_truth)[pos]),
        "negmass_fit": twin.negative_mass_ratio(mesh, fit["c"]),
    }
    return out, z_fit, solver


def headline(mesh, K_truth, K_model, bc, times, sD_true=1.1, r_true=0.0135, phys=twin.Physics(),
             x0=(1.0, 0.018), lb=(0.6, 0.0), ub=(1.4, 0.036), method="cg", tmask=None):
    clock = twin.timer()
    truth_solver = twin.Solver(mesh, K_truth, bc, times, phys=phys, method=method)
    c_truth = truth_solver.run(sD_true, r_true)
    z_truth = truth_solver.weighted(c_truth)
    out, z_fit, solver = fit_and_score(mesh, K_model, bc, times, z_truth, phys=phys, x0=x0, lb=lb, ub=ub,
                                       method=method, tmask=tmask)
    out["negmass_truth"] = twin.negative_mass_ratio(mesh, c_truth)
    out["min_c_truth"] = float(c_truth.min())
    out["seconds"] = round(clock(), 1)
    roi = twin.TissueROI(mesh)
    curves = {"times_h": times, "truth_roi_mean": (roi.mean_op @ c_truth.T).T,
              "fit_roi_mean": (roi.mean_op @ (z_fit / np.sqrt(mesh.w / mesh.V)).T).T}
    return out, curves, z_truth, z_fit
