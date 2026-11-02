"""Q1 error budget: how much apparent diffusion enhancement can modelling and measurement errors create?

A truth is generated with a known transport model; an analyst then fits the enhancement factor
alpha (= s_D, the multiplier on the analyst's diffusion field) and the clearance r the way human
tracer-MRI studies do, with one or more realistic errors. With the analyst using the true diffusion
field, the true alpha is the truth's s_D, so alpha_fit / alpha_true - 1 is the apparent enhancement
the errors create.

Usage: python q1_error_budget.py <case> | list
Writes results/nd/q1/<case>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nd_common as nd  # noqa: E402
import twin  # noqa: E402

OUT = nd.ND_OUT / "q1"
TRUTH_CACHE = nd.ND_OUT / "truths"
NOISE_SIGMA = 0.02          # mmol/L per node, estimated from the deep pre-arrival Gonzo values (q4)

TRUTHS = {
    "base": dict(field="tensor", sD=1.0, r=0.018),
    "r0": dict(field="tensor", sD=1.0, r=0.0),
    "vinje": dict(field="tensor", sD=3.5, r=0.186),
}
for F in ("pial_inflow_drainage", "pial_to_ventricle_flux", "basal_to_dorsal"):
    for v in (1.0, 2.0, 4.0):
        TRUTHS[f"flow_{F}_{v:g}"] = dict(field="tensor", sD=1.0, r=0.018, flow=F, vs=v)

PERFECT = dict(field="tensor", bc="curve", amp=1.0, obs="roi", window=None, pv=None, noise=0.0, fix_r=None,
               scans_only=True)
COMBO = dict(bc="linear", amp=0.85, pv=(0.5, 1.0), noise=NOISE_SIGMA)


def case(truth="base", **analyst):
    return {"truth": truth, "analyst": dict(PERFECT, **analyst)}


CASES = {
    "A0_perfect": case(),
    "A1_linear_bc": case(bc="linear"),
    "A2_amp0.7": case(amp=0.7), "A2_amp0.85": case(amp=0.85), "A2_amp1.15": case(amp=1.15),
    "A2_amp1.3": case(amp=1.3),
    "A3_pv0.25": case(pv=(0.25, 1.0)), "A3_pv0.5": case(pv=(0.5, 1.0)), "A3_pv0.5_ell2": case(pv=(0.5, 2.0)),
    "A4_noise": case(noise=NOISE_SIGMA),
    "A5_win24": case(window=24.0), "A5_dense": case(scans_only=False),
    "A6_fix_r0": case(fix_r=0.0),
    "A6_fix_r0_win24": case(fix_r=0.0, window=24.0),
    "A7_field_perfect": case(obs="field"), "A7_field_linear_bc": case(obs="field", bc="linear"),
    "A7_field_amp0.85": case(obs="field", amp=0.85), "A7_field_pv0.5": case(obs="field", pv=(0.5, 1.0)),
    "A7_field_noise": case(obs="field", noise=NOISE_SIGMA),
    "B1_literature_field": case(field="literature"), "B2_dataiso_field": case(field="dataiso"),
    "B1_literature_field_obsfield": case(field="literature", obs="field"),
    "C1_combo": case(**COMBO), "C1_combo_field": case(obs="field", **COMBO),
    "C1_combo_literature": case(field="literature", **COMBO),
    "C1_combo_win24_fix_r0": case(window=24.0, fix_r=0.0, **COMBO),
    "D_r0_perfect": case("r0"), "D_r0_combo": case("r0", **COMBO),
    "D_vinje_perfect": case("vinje"), "D_vinje_combo": case("vinje", **COMBO),
    "D_vinje_combo_field": case("vinje", obs="field", **COMBO),
}
for key in TRUTHS:
    if key.startswith("flow_"):
        CASES[f"E_{key}"] = case(key)
        CASES[f"E_{key}_field"] = case(key, obs="field")
        if key.endswith("_2"):
            CASES[f"E_{key}_combo"] = case(key, **COMBO)


def field_of(S, name):
    return {"tensor": S.fields["DataTensor"], "literature": S.fields["LiteratureIsotropic"],
            "dataiso": S.fields["DataIsotropic"]}[name]


def truth_trajectory(S, key):
    """Noiseless truth concentration (nt,n), cached as float32."""
    TRUTH_CACHE.mkdir(parents=True, exist_ok=True)
    f = TRUTH_CACHE / f"{key}.npy"
    if f.exists():
        return np.load(f).astype(float)
    spec = TRUTHS[key]
    K = S.mesh.stiffness(field_of(S, spec["field"]))
    bc = twin.Boundary(S.params)
    if spec.get("flow"):
        v = np.load(nd.ND_OUT / f"velocity_{spec['flow']}.npy")
        L, q_in, _ = nd.advection_operator(S.mesh, v)
        solver = nd.GenSolver(S.mesh, K, bc, S.t, L=L, q_in=q_in)
        c, _ = solver.run(spec["sD"], spec["r"], spec["vs"])
    else:
        c, _ = nd.GenSolver(S.mesh, K, bc, S.t).run(spec["sD"], spec["r"])
    np.save(f, c.astype(np.float32))
    return c


def run(name):
    spec = CASES[name]
    a = spec["analyst"]
    tspec = TRUTHS[spec["truth"]]
    S = nd.Setup()
    clock = twin.timer()
    c_true = truth_trajectory(S, spec["truth"])
    bc_true = twin.Boundary(S.params)
    Bt_true = np.stack([bc_true(t) for t in S.t])

    # what the analyst observes
    c_obs = c_true
    if a["pv"] is not None:
        c_obs = nd.partial_volume(S, c_obs, Bt_true, f0=a["pv"][0], ell=a["pv"][1])
    if a["noise"] > 0:
        c_obs = nd.add_noise(c_obs, a["noise"], seed=20261101)

    # the analyst's model
    bc_model = bc_true if a["bc"] == "curve" else nd.sampled_linear(bc_true)
    if a["amp"] != 1.0:
        bc_model = nd.ScaledBoundary(bc_model, a["amp"])
    K = S.mesh.stiffness(field_of(S, a["field"]))
    solver = nd.GenSolver(S.mesh, K, bc_model, S.t)
    obs = S.roi if a["obs"] == "roi" else nd.field_obs
    tmask = S.tmask(max_h=a["window"], scans_only=a["scans_only"])
    y = obs(solver.weighted(c_obs))
    if a["fix_r"] is None:
        res = nd.fit(solver, obs, y, tmask, names=("sD", "r"), starts=((1.0, 0.02), (3.0, 0.15)))
    else:
        res = nd.fit(solver, obs, y, tmask, names=("sD",), starts=((1.0,), (3.0,)), fixed={"r": a["fix_r"]})
    c_fit = res.pop("c")
    sc = nd.score(S, c_fit, c_true, S.tmask())
    out = {"case": name, "truth": spec["truth"], "truth_spec": tspec, "analyst": a,
           "alpha_fit": res["sD"], "r_fit": res["r"], "alpha_true": tspec["sD"], "r_true": tspec["r"],
           "apparent_enhancement": res["sD"] / tspec["sD"] - 1.0, "fit": res, **sc,
           "truth_roi_means": nd.roi_means(S, c_true)[S.scan_idx].tolist(),
           "fit_roi_means": nd.roi_means(S, c_fit)[S.scan_idx].tolist(),
           "seconds": round(clock(), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{name}.json", "w"), indent=2, default=float)
    print(json.dumps({k: out[k] for k in ("case", "alpha_fit", "r_fit", "apparent_enhancement", "roi_L2",
                                          "native_L2", "hidden_fraction", "seconds")}, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(CASES))
    elif sys.argv[1] == "truths":
        S = nd.Setup()
        truth_trajectory(S, sys.argv[2])
    else:
        run(sys.argv[1])
