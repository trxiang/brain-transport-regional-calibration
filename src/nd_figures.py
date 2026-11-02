"""Figures for the new-directions studies (Q1 error budget, Q4 real-data fits, fingerprint).

Usage: python nd_figures.py        reads results/nd/*.csv/json, writes results/nd/figures/*.png|pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import twin  # noqa: E402

ND = twin.REPO / "results/nd"
FIG = ND / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

Q1_ROWS = [  # (label, roi case, field case)
    ("Perfect analyst", "A0_perfect", "A7_field_perfect"),
    ("Noise 0.02 mmol/L", "A4_noise", "A7_field_noise"),
    ("CSF input linear in time", "A1_linear_bc", "A7_field_linear_bc"),
    ("CSF input 15% low", "A2_amp0.85", "A7_field_amp0.85"),
    ("CSF input 15% high", "A2_amp1.15", None),
    ("Partial volume, 1 mm", "A3_pv0.5", "A7_field_pv0.5"),
    ("Partial volume, 2 mm", "A3_pv0.5_ell2", None),
    ("Clearance fixed at 0", "A6_fix_r0", None),
    ("Literature diffusion field", "B1_literature_field", "B1_literature_field_obsfield"),
    ("Isotropic DTI field", "B2_dataiso_field", None),
    ("Flow 1 µm/min (drainage)", "E_flow_pial_inflow_drainage_1", "E_flow_pial_inflow_drainage_1_field"),
    ("Flow 2 µm/min (drainage)", "E_flow_pial_inflow_drainage_2", "E_flow_pial_inflow_drainage_2_field"),
    ("Flow 4 µm/min (drainage)", "E_flow_pial_inflow_drainage_4", "E_flow_pial_inflow_drainage_4_field"),
    ("Flow 4 µm/min (basal-dorsal)", "E_flow_basal_to_dorsal_4", "E_flow_basal_to_dorsal_4_field"),
    ("Combined input + PV errors", "C1_combo", "C1_combo_field"),
]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_q1():
    df = pd.read_csv(ND / "q1_summary.csv").set_index("case")
    rows = [r for r in Q1_ROWS if r[1] in df.index]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(rows) + 1.0))
    for yi, (lab, roi, fld) in zip(y, rows):
        ax.barh(yi + 0.18, df.loc[roi, "alpha_fit"], 0.34, color="#c0504d", label="tissue curves" if yi == y[0] else None)
        if fld and fld in df.index:
            ax.barh(yi - 0.18, df.loc[fld, "alpha_fit"], 0.34, color="#4f81bd",
                    label="native field" if yi == y[0] else None)
    ax.axvline(1.0, color="k", lw=0.8)
    ax.axvline(3.5, color="0.5", lw=0.8, ls="--")
    ax.text(3.55, y[0] + 0.6, "Vinje et al. alpha = 3.5", color="0.4", fontsize=7)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("fitted enhancement alpha (truth: alpha = 1)")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 3.5, 5], ["0.5", "1", "2", "3.5", "5"])
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "q1_apparent_enhancement")


def fig_q4():
    df = pd.read_csv(ND / "q4_summary.csv")
    d = df[df.model == "diff"].copy()
    fields = ["tensor", "literature", "dataiso"]
    combos = [(bc, obs, w) for bc in ("curve", "linear") for obs in ("roi", "depth", "field") for w in ("all", "24")]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    marks = {"tensor": "o", "literature": "s", "dataiso": "^"}
    for f in fields:
        xs, ys = [], []
        for i, (bc, obs, w) in enumerate(combos):
            s = d[(d.field == f) & (d.bc == bc) & (d.obs == obs) & (d.window == w)]
            if len(s):
                xs.append(i + {"tensor": -0.2, "literature": 0, "dataiso": 0.2}[f])
                ys.append(s.alpha_fit.iloc[0])
        ax.scatter(xs, ys, marker=marks[f], s=22, zorder=3,
                   label={"tensor": "DTI tensor field", "literature": "literature field", "dataiso": "isotropic DTI field"}[f])
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhline(3.5, color="0.5", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 3.5, 5, 9], ["0.5", "1", "2", "3.5", "5", "9"])
    ax.set_xticks(range(len(combos)), [f"{ {'roi': 'curves'}.get(obs, obs)}\n{'≤24 h' if w == '24' else 'all'}"
                                       for bc, obs, w in combos],
                  fontsize=7)
    for k, bc in enumerate(("curve-fit CSF input", "linear CSF input")):
        ax.text(2.5 + 6 * k, ax.get_ylim()[1] * 0.95, bc, ha="center", va="top", fontsize=8)
    ax.axvline(5.5, color="0.8", lw=0.8)
    ax.set_ylabel("fitted alpha (real data)")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, "q4_alpha_by_choice")


SIG_LABELS = {"A1_linear_bc": "CSF input linear in time", "A2_amp0.85": "CSF input 15% low",
              "A2_amp1.15": "CSF input 15% high", "A3_pv0.5": "Partial volume, 1 mm",
              "A3_pv0.5_ell2": "Partial volume, 2 mm", "A6_fix_r0": "Clearance fixed at 0",
              "B1_literature_field": "Literature diffusion field", "C1_combo": "Combined input + PV errors"}


def sig_label(k):
    if k in SIG_LABELS:
        return SIG_LABELS[k]
    _, _, rest = k.partition("E_flow_")
    pat, _, v = rest.rpartition("_")
    return f"Flow {v} µm/min ({pat.replace('pial_inflow_drainage', 'drainage').replace('pial_to_ventricle_flux', 'pial-ventricle').replace('basal_to_dorsal', 'basal-dorsal')})"


def fig_fingerprint():
    fp = json.load(open(ND / "fingerprint.json"))
    job = "tensor__curve__roi__all__diff"
    R = np.array(fp["real"][job])                         # (nses, nrow)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.8), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.9})
    v = np.abs(R).max()
    im = a1.imshow(R.T, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    a1.set_xticks(range(len(fp["sessions_h"])), [f"{t:.0f} h" for t in fp["sessions_h"]])
    a1.set_yticks(range(len(fp["rows"])), [r.replace("0-1e-06 mm", "surface").replace("1e-06-", "0-")
                                           .replace("-inf", "+") for r in fp["rows"]], fontsize=7)
    a1.set_title("real data minus best diffusion fit", fontsize=9)
    fig.colorbar(im, ax=a1, label="residual (mmol/L)", orientation="horizontal", shrink=0.9, pad=0.1)
    names = sorted(fp["cosine"], key=lambda k: fp["cosine"][k][job])
    a2.barh(range(len(names)), [fp["cosine"][k][job] for k in names],
            color=["#4f81bd" if fp["cosine"][k][job] > 0 else "#c0504d" for k in names])
    a2.set_yticks(range(len(names)), [sig_label(k) for k in names], fontsize=7)
    a2.axvline(0, color="k", lw=0.8)
    a2.set_xlim(-1, 1)
    a2.set_xlabel("cosine similarity with the real misfit")
    save(fig, "fingerprint")


def fig_profile():
    """Real-data profile of alpha: chi^2 rise above its minimum, scaled by the misfit per observation."""
    df = pd.read_csv(ND / "q4_profile.csv")
    nobs = {"roi": 12, "depth": 56, "field": 4 * 99755}
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.8), sharex=True, sharey=True)
    for ax, obs in zip(axs, ("roi", "depth", "field")):
        for bc, col in (("curve", "#4f81bd"), ("linear", "#c0504d")):
            s = df[(df.obs == obs) & (df.bc == bc)].sort_values("alpha")
            c = s[f"chi2_{obs}"].to_numpy()
            ax.plot(s.alpha, (c - c.min()) / (c.min() / nobs[obs]), "o-", ms=3, color=col,
                    label=f"{bc} CSF input")
        ax.axhline(3.84, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(-0.2, 1e5)
        ax.set_yticks([0, 1, 3.84, 10, 100, 1e3, 1e4], ["0", "1", "3.84", "10", "100", "1e3", "1e4"])
        ax.set_xticks([0.3, 0.5, 1, 2, 3.5, 8], ["0.3", "0.5", "1", "2", "3.5", "8"])
        ax.set_title({"roi": "tissue curves", "depth": "depth profiles", "field": "native field"}[obs], fontsize=9)
        ax.set_xlabel("alpha (r refitted)")
    axs[0].set_ylabel("scaled chi2 rise above minimum")
    axs[0].legend(frameon=False, fontsize=7)
    save(fig, "q4_alpha_profile")


OBS_LABELS = {"roi": "tissue curves", "depth": "depth profiles", "block8": "8 mm blocks", "field": "native field"}


def fig_q2():
    df = pd.read_csv(ND / "q2_summary.csv")
    df = df[df.flow_model.notna()]
    pats = [("pial_inflow_drainage", "drainage"), ("pial_to_ventricle_flux", "pial-ventricle"),
            ("basal_to_dorsal", "basal-dorsal")]
    marks = {"roi": "o", "depth": "s", "block8": "D", "field": "^"}
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.9), sharey=True)
    for ax, (p, lab) in zip(axs, pats):
        s = df[(df.flow_model == p) & ((df.flow_true == p) | (df.flow_true == "none"))]
        for (err, obs), g in s.groupby(["errors", "obs"]):
            x = g.speed_true.to_numpy() * (1 + 0.06 * list(marks).index(obs))
            ax.scatter(np.maximum(x, 0.2), g.vs_flow, marker=marks[obs], s=20,
                       facecolors="#4f81bd" if err == "exact" else "none",
                       edgecolors="#4f81bd" if err == "exact" else "#c0504d",
                       label=(f"{OBS_LABELS[obs]}, {'exact inputs' if err == 'exact' else 'realistic errors'}"
                              if p == pats[0][0] else None))
        ax.plot([0.4, 5], [0.4, 5], color="0.6", lw=0.8)
        ax.axhline(20, color="0.8", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xticks([0.2, 0.5, 1, 2, 4], ["0", "0.5", "1", "2", "4"])
        ax.set_title(f"flow pattern: {lab}", fontsize=9)
        ax.set_xlabel("true speed (µm/min)")
    axs[0].set_ylabel("fitted speed (µm/min)")
    fig.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "q2_flow_speed_recovery")


def fig_q5():
    df = pd.read_csv(ND / "q5_summary.csv")
    obs = [("roi", "tissue curves"), ("depth", "depth profiles"), ("field", "native field")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.1), gridspec_kw={"wspace": 0.35})
    sets = [("base", "(alpha, r) only", "#c0504d"), ("amp", "+ input scale", "#9bbb59"),
            ("amp_pv", "+ input scale + partial volume", "#4f81bd"), ("all", "+ exchange as well", "#8064a2")]
    for ax, kind, cases in ((a1, "syn", [("base", 1.0), ("vinje", 3.5)]), (a2, "real", [("curve", None), ("linear", None)])):
        s = df[df.kind == kind]
        x0 = 0
        ticks, labs = [], []
        for case, truth in cases:
            for o, olab in obs:
                for j, (st, slab, col) in enumerate(sets):
                    g = s[(s.case == case) & (s.obs == o) & (s.set == st)]
                    if len(g):
                        ax.scatter(x0 + 0.15 * (j - 1.5), g.sD.iloc[0], color=col, s=22, zorder=3,
                                   label=slab if (case, o) == (cases[0][0], "roi") else None)
                if truth:
                    ax.plot([x0 - 0.4, x0 + 0.4], [truth, truth], color="k", lw=1.2)
                ticks.append(x0)
                labs.append(olab.replace(" ", "\n"))
                x0 += 1
            x0 += 0.5
        ax.set_xticks(ticks, labs, fontsize=7)
        ax.set_yscale("log")
        ax.set_yticks([0.5, 1, 2, 3.5, 5, 10], ["0.5", "1", "2", "3.5", "5", "10"])
        ax.axhline(1.0, color="0.7", lw=0.6, ls=":")
        ax.set_ylabel("fitted enhancement alpha")
    a1.set_title("synthetic (black bars: true alpha)\ntruth alpha = 1 (left), 3.5 (right)", fontsize=8)
    a2.set_title("Gonzo data\nCSF curve fit (left), linear interpolation (right)", fontsize=8)
    handles, labels = a2.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "q5_nuisance_fits")


def fig_q6():
    df = pd.read_csv(ND / "q6_summary.csv")
    truths = [("flow_pial_inflow_drainage_0.5", "drainage 0.5"), ("flow_pial_inflow_drainage_1", "drainage 1"),
              ("flow_pial_to_ventricle_flux_1", "pial-ventricle 1"), ("flow_basal_to_dorsal_1", "basal-dorsal 1")]
    obs = [("roi", "tissue curves"), ("depth", "depth profiles"), ("block8", "8 mm blocks")]
    cols = ["#9bbb59", "#4f81bd", "#c0504d", "#8064a2"]
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.8), sharey=True)
    for ax, (o, olab) in zip(axs, obs):
        for (t, tlab), col in zip(truths, cols):
            g = df[(df.truth == t) & (df.obs == o)].sort_values("ell_mm")
            if not len(g):
                continue
            ax.plot([0.0] + g.ell_mm.tolist(), [g.lambda_iid.iloc[0]] + g.lambda_corr.tolist(), "o-", color=col,
                    ms=3.5, lw=1, label=tlab + " µm/min" if o == "roi" else None)
        ax.axhline(7.85, color="0.5", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xticks([0, 2.5, 5, 10], ["iid", "2.5", "5", "10"])
        ax.set_xlabel("noise correlation length (mm)")
        ax.set_title(olab, fontsize=9)
    axs[0].text(10, 8.6, "80% power", ha="right", fontsize=7, color="0.4")
    axs[0].set_ylabel("noncentrality (flow vs diffusion)")
    fig.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "q6_correlated_noise")


def fig_robust():
    g = pd.read_csv(ND / "geometry_compare.csv")
    g = g[(g.study == "Q1") & g.vinje14.notna() & g.vinje4.notna()]
    mc = json.load(open(ND / "mesh_check.json"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.4), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.6})
    y = np.arange(len(g))[::-1]
    for k, (col, lab, m) in enumerate((("gonzo", "Gonzo", "o"), ("vinje14", "Vinje mesh 14", "s"),
                                       ("vinje4", "Vinje mesh 4", "D"))):
        a1.scatter(g[col], y + 0.2 * (1 - k), marker=m, s=20, label=lab, zorder=3)
    names = {r[1]: r[0] for r in Q1_ROWS}
    names.update({r[2]: r[0] + ", field" for r in Q1_ROWS if r[2]})
    names.update({"C1_combo_field": "Combined errors, native field", "C1_combo": "Combined errors, tissue curves"})
    a1.set_yticks(y, [names.get(c, c) for c in g.case], fontsize=7)
    a1.set_xscale("log")
    a1.set_xticks([0.5, 1, 2, 3.5, 5], ["0.5", "1", "2", "3.5", "5"])
    a1.axvline(1.0, color="k", lw=0.8)
    a1.set_xlabel("fitted alpha (truth alpha = 1)")
    a1.legend(frameon=False, fontsize=7, loc="lower right")
    a1.set_title("same errors, three brains", fontsize=9)
    keys = [k for k in mc if k != "base"]
    labs = {"flow_pial_inflow_drainage_1": "drainage 1", "flow_pial_inflow_drainage_4": "drainage 4",
            "flow_pial_to_ventricle_flux_4": "pial-vent. 4", "flow_basal_to_dorsal_4": "basal-dors. 4"}
    x = np.arange(len(keys))
    for j, (o, mk) in enumerate((("roi", "o"), ("field", "^"))):
        coarse = [mc[k][f"alpha_coarse_{o}"] / mc["base"][f"alpha_coarse_{o}"] for k in keys]
        fine = [mc[k][f"alpha_fine_{o}"] / mc["base"][f"alpha_fine_{o}"] for k in keys]
        a2.scatter(x + 0.12 * (2 * j - 1) - 0.04, coarse, marker=mk, facecolors="none", edgecolors="#c0504d",
                   label=f"{o}, base mesh")
        a2.scatter(x + 0.12 * (2 * j - 1) + 0.04, fine, marker=mk, color="#4f81bd", label=f"{o}, refined mesh")
    a2.set_xticks(x, [labs.get(k, k) for k in keys], fontsize=7, rotation=20)
    a2.axhline(1.0, color="0.7", lw=0.6, ls=":")
    a2.set_ylabel("alpha / alpha(no flow, same mesh)")
    a2.set_title("flow-induced enhancement vs mesh (µm/min)", fontsize=9)
    a2.legend(frameon=False, fontsize=6.5, loc="upper right")
    save(fig, "robustness_geometry_mesh")


STEP_LABELS = {"S0": "exact inputs", "S1": "+ combined errors", "S2": "+ scans <= 48 h", "S3": "+ regional isotropic D",
               "S4": "+ alpha in cortex only", "S5": "+ boundary voxel prescribed", "S6": "+ alpha >= 1 (published)",
               "P_exact": "published, exact data", "P_nopv": "published, no partial volume", "P_roi": "published, tissue curves"}


def fig_pipeline():
    df = pd.read_csv(ND / "q7_summary.csv")
    steps = [s for s in STEP_LABELS if s in set(df.step)]
    fig, (ax, a2) = plt.subplots(1, 2, figsize=(10.0, 0.3 * len(steps) + 1.4),
                                 gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.35})
    y = np.arange(len(steps))[::-1]
    series = [("syn", "base", "truth alpha = 1", "#4f81bd", "o"), ("syn", "vinje", "truth alpha = 3.5", "#c0504d", "s"),
              ("real", "linear", "Gonzo data (linear input)", "k", "D"), ("real", "curve", "Gonzo data (curve input)", "0.5", "^")]
    for j, (kind, case, lab, col, mk) in enumerate(series):
        g = df[(df.kind == kind) & (df.case == case)].set_index("step")
        xs = [g.alpha.get(s, np.nan) for s in steps]
        ax.plot(xs, y + 0.12 * (1.5 - j), mk, color=col, ms=5, label=lab, ls="none")
    ax.axvline(1.0, color="#4f81bd", lw=0.7, ls=":")
    ax.axvline(3.5, color="#c0504d", lw=0.7, ls=":")
    ax.set_yticks(y, [STEP_LABELS[s] for s in steps], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 3.5, 5, 10], ["0.5", "1", "2", "3.5", "5", "10"])
    ax.set_xlabel("fitted alpha")
    ax.set_title("(a) published choices switched on one at a time", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.4, 1.06))
    truth_alpha = {"base": 1.0, "cx2": 2.0, "cx3.5": 3.5}           # alpha in cortex (base: alpha 1 everywhere)
    syn = df[df.kind == "syn"]
    for step, lab, col, mk in (("S6", "combined errors", "#c0504d", "o"),
                               ("S6w", "same, volume-weighted", "#c0504d", "x"),
                               ("P_nopv", "no partial volume", "#9bbb59", "s"),
                               ("P_exact", "exact observations", "k", "^")):
        g = syn[(syn.step == step) & syn.case.isin(truth_alpha)]
        if len(g):
            g = g.assign(t=g.case.map(truth_alpha)).sort_values("t")
            a2.plot(g.t, g.alpha, marker=mk, color=col, ls="-" if mk != "x" else "none", lw=0.8, ms=5, label=lab)
    for step, lab, ls in (("S6", " Gonzo", "--"), ("S6w", " Gonzo (vol.)", ":")):
        real = df[(df.kind == "real") & (df.case == "linear") & (df.step == step)]
        if len(real):
            a2.axhline(real.alpha.iloc[0], color="0.3", lw=0.8, ls=ls)
            a2.text(3.55, real.alpha.iloc[0], lab, va="center", fontsize=6.5, color="0.3")
    a2.axhline(3.5, color="#c0504d", lw=0.7, ls="--")
    a2.text(3.55, 3.5, " published", va="center", fontsize=6.5, color="#c0504d")
    a2.plot([1, 3.5], [1, 3.5], color="0.7", lw=0.7, ls=":")
    a2.set_xlabel("true alpha in cortex")
    a2.set_ylabel("alpha returned")
    a2.set_xticks([1, 2, 3.5])
    a2.set_title("(b) how the pipeline maps the truth", fontsize=8.5)
    a2.legend(frameon=False, fontsize=6.5, loc="upper left")
    save(fig, "q7_published_pipeline")

PERF_LABELS = {"perf_Q15_l1": "15 mL/h\n1 mm", "perf_Q15_l3": "15 mL/h\n3 mm", "perf_Q45_l1": "45 mL/h\n1 mm",
               "perf_Q45_l3": "45 mL/h\n3 mm"}


def fig_perfusion():
    df = pd.read_csv(ND / "q8_summary.csv")
    j = df.job.str.split("__", expand=True)
    syn = df[j[0] == "syn"].assign(truth=j[1], err=j[2])
    real = df[j[0] == "real"].assign(bc=j[1], ell=j[4].astype(float, errors="ignore") if 4 in j else None)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.0, 3.0), gridspec_kw={"wspace": 0.75, "width_ratios": [1.1, 1, 1.0]})
    truths = list(PERF_LABELS)
    x = np.arange(len(truths))
    for k, (err, o, mk, col, lab) in enumerate((("exact", "roi", "o", "0.4", "exact, tissue curves"),
                                                ("exact", "field", "o", "k", "exact, native field"),
                                                ("combo", "field", "^", "#c0504d", "combined errors, field"))):
        g = syn[(syn.err == err) & (syn.obs == o) & (syn.model == "diff")].set_index("truth")
        a1.scatter(x + 0.14 * (k - 1), [g.sD.get(t, np.nan) for t in truths], marker=mk, s=22,
                   facecolors="none" if o == "roi" else col, edgecolors=col, label=lab, zorder=3)
    a1.axhline(1.0, color="k", lw=0.8)
    a1.set_yscale("log")
    a1.set_yticks([1, 2, 3.5, 5, 10], ["1", "2", "3.5", "5", "10"])
    a1.set_xticks(x, [PERF_LABELS[t] for t in truths], fontsize=7)
    a1.set_ylabel("apparent alpha (diffusion-only fit)")
    a1.set_title("truth: alpha = 1 plus CSF perfusion", fontsize=8.5)
    a1.legend(frameon=False, fontsize=6.5, loc="upper left")
    tq = {"base": 0.0, **{t: float(t.split("_Q")[1].split("_")[0]) for t in truths}}
    for k, (o, mk, col) in enumerate((("depth", "s", "#9bbb59"), ("field", "^", "#c0504d"))):
        g = syn[(syn.err == "combo") & (syn.obs == o) & (syn.model == "perf")]
        a2.scatter([tq[t] + 1.5 * (k - 0.5) for t in g.truth], g.Q, marker=mk, color=col, s=22,
                   label=f"combined errors, {o}", zorder=3)
    g = syn[(syn.err == "exact") & (syn.model == "perf")]
    a2.scatter([tq[t] for t in g.truth], g.Q, marker="o", color="k", s=14, label="exact data, field", zorder=4)
    a2.plot([0, 50], [0, 50], color="0.6", lw=0.8, ls="--")
    a2.axhspan(25, 65, color="#4f81bd", alpha=0.12, lw=0)
    a2.set_xlabel("true perfusion Q (mL/h)")
    a2.set_ylabel("fitted Q (mL/h), alpha fixed at 1")
    a2.set_title("perfusion is not identified\n(shaded: 45 +- 20 mL/h, Wahlin 2026)", fontsize=8.5)
    a2.legend(frameon=False, fontsize=6.5, loc="upper left")
    r = df[(j[0] == "real") & (j[3] == "perf") & (j[2] != "roi")].copy()
    r["lab"] = j[1] + ", " + j[2] + ", " + j[4] + " mm" + np.where(j[5] == "amp", ", scale free", "")
    r = r.sort_values("Q")
    a3.barh(np.arange(len(r)), r.Q, color=np.where(r.nuis == "amp", "#4f81bd", "0.6"), height=0.7)
    a3.set_yticks(np.arange(len(r)), r.lab, fontsize=6)
    for yi, q in enumerate(r.Q):
        a3.text(q + 0.8, yi, f"{q:.0f}", va="center", fontsize=5.5)
    a3.axvspan(25, 65, color="#4f81bd", alpha=0.12, lw=0)
    a3.set_xlabel("fitted Q (mL/h), alpha = 1")
    a3.set_title("Gonzo data (blue: input scale fitted)", fontsize=8.5)
    save(fig, "q8_perfusion")


GRP_LABELS = {"csfslow": "CSF slower", "csffast": "CSF faster", "rlow": "r -33%", "rlow_csfslow": "r -33% +\nCSF slower"}
ANALYST_STYLE = {"exact_field": ("exact inputs", "k", "o"), "combo_roi": ("errors, tissue curves", "#c0504d", "s"),
                 "combo_field": ("errors, native field", "#4f81bd", "^"), "vinje": ("published pipeline", "#9bbb59", "D")}


def fig_groups():
    dd = pd.read_csv(ND / "q9_summary.csv")
    dd["lvl"] = dd.truth.str[0]
    dd["grp"] = dd.truth.str[2:]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.0), sharex="col", gridspec_kw={"hspace": 0.15, "wspace": 0.3})
    for c, (lvl, title) in enumerate((("a", "slow clearance (r = 0.018 1/h)"), ("b", "fast clearance (r = 0.12 1/h)"))):
        g0 = dd[dd.lvl == lvl]
        grps = [k for k in GRP_LABELS if k in set(g0.grp)]
        x = np.arange(len(grps))
        for k, (an, (lab, col, mk)) in enumerate(ANALYST_STYLE.items()):
            g = g0[g0.analyst == an].set_index("grp")
            xs = x + 0.13 * (k - 1.5)
            axes[0, c].scatter(xs, [g.dalpha_pct.get(t, np.nan) for t in grps], color=col, marker=mk, s=22,
                               label=lab, zorder=3)
            dr = np.array([g.dr_pct_fit.get(t, np.nan) if g.r_ref.get(t, 0) > 1e-3 else np.nan for t in grps])
            axes[1, c].scatter(xs, dr, color=col, marker=mk, s=22, zorder=3)
            for xi, t in zip(xs, grps):
                if t in g.index and g.r_ref[t] <= 1e-3:
                    axes[1, c].text(xi, 5, "r=0", color=col, fontsize=5.5, ha="center", rotation=90)
        true_dr = [-100 / 3 if t.startswith("rlow") else 0.0 for t in grps]
        axes[1, c].scatter(x, true_dr, marker="_", s=300, color="k", lw=1.5, label="true difference", zorder=2)
        axes[0, c].axhline(0, color="k", lw=1.0)
        axes[0, c].set_title(title, fontsize=9)
        axes[1, c].set_xticks(x, [GRP_LABELS[t] for t in grps], fontsize=7.5)
        axes[0, c].set_ylim(-35, 45)
        axes[1, c].set_ylim(-50, 20)
    axes[0, 0].set_ylabel("fitted alpha difference (%)\n(true: 0)")
    axes[1, 0].set_ylabel("fitted r difference (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.94))
    axes[1, 0].legend(frameon=False, fontsize=6.5, loc="lower left")
    save(fig, "q9_groups")


def fig_realdata():
    """Figure 6: the three real-data panels tiled (a, b top; c bottom) from their rendered PNGs."""
    ims = [plt.imread(FIG / f"{n}.png") for n in ("q4_alpha_by_choice", "q4_alpha_profile", "fingerprint")]
    fig = plt.figure(figsize=(10.0, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[ims[0].shape[1] / ims[0].shape[0],
                                                                          ims[1].shape[1] / ims[1].shape[0]],
                          hspace=0.05, wspace=0.02)
    for k, (im, pos) in enumerate(zip(ims, (gs[0, 0], gs[0, 1], gs[1, :]))):
        ax = fig.add_subplot(pos)
        ax.imshow(im)
        ax.set_axis_off()
        ax.text(0.0, 1.0, "abc"[k], transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    save(fig, "real_data_composite")


if __name__ == "__main__":
    fig_perfusion()
    fig_groups()
    fig_pipeline()
    fig_q5()
    fig_q6()
    fig_robust()
    fig_profile()
    fig_q2()
    fig_q1()
    fig_q4()
    fig_fingerprint()
    fig_realdata()
    print("figures in", FIG)
