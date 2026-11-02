"""Journal-format figures for the JCBFM submission (at most 170 mm wide, EPS + 600 dpi TIFF).

Re-renders the nd_figures panels with the figure width capped so that text stays legible at print size,
lays out the perfusion figure for a single page width, and builds the real-data figure as a vector composite.
Usage: python jcbfm_figures.py <outdir>
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import nd_figures as nf  # noqa: E402

CAP_IN = 6.5                                   # figure width cap before the tight bounding box (inches)
MAX_MM = 170.0                                 # journal maximum figure width
MAX_H_MM = 234.0                               # journal maximum figure height
FONT = os.environ.get("FIG_FONT")              # e.g. Arial for Springer journals
PREFIX = os.environ.get("FIG_PREFIX", "Figure")
MIN_PT = float(os.environ.get("FIG_MIN_PT", "0"))   # minimum lettering size before export scaling (0 = off)
SYMBOLS = [(r"\balpha\b", "α"), (r"\bchi2\b", "χ²"), (r"\+-", "±"), (r"<=", "≤"), (r">=", "≥"), (r"Wahlin", "Wåhlin")]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "jcbfm_figs")
PDF = OUT / "pdf"

_subplots, _figure = plt.subplots, plt.figure


def _cap(kw):
    fs = kw.get("figsize")
    if fs and fs[0] > CAP_IN:
        kw["figsize"] = (CAP_IN, fs[1] * max(CAP_IN / fs[0], 0.8))
    return kw


def capped_subplots(*a, **kw):
    return _subplots(*a, **_cap(kw))


plt.subplots = capped_subplots
plt.figure = lambda *a, **kw: _figure(*a, **_cap(kw))


def polish(fig):
    """Typeset symbols (alpha -> α, chi2 -> χ², <= -> ≤) and raise lettering to at least MIN_PT."""
    from matplotlib.text import Text
    fig.canvas.draw()
    for t in fig.findobj(Text):
        s = t.get_text()
        if s and "$" not in s:
            for pat, rep in SYMBOLS:
                s = re.sub(pat, rep, s)
            t.set_text(s)
        if MIN_PT and t.get_fontsize() < MIN_PT:
            t.set_fontsize(MIN_PT)
    from matplotlib.ticker import FixedFormatter
    for ax in fig.axes:                                  # fixed tick labels are redrawn from their formatter
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, FixedFormatter):
                seq = []
                for lab in fmt.seq:
                    for pat, rep in SYMBOLS:
                        lab = re.sub(pat, rep, lab) if "$" not in lab else lab
                    seq.append(lab)
                fmt.seq = seq
    if MIN_PT:
        for ax in fig.axes:
            for which, labels in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
                if labels and min(l.get_fontsize() for l in labels) < MIN_PT:
                    ax.tick_params(axis=which, labelsize=MIN_PT)


def save(fig, name):
    PDF.mkdir(parents=True, exist_ok=True)
    if MIN_PT or os.environ.get("FIG_SYMBOLS"):
        polish(fig)
    fig.savefig(PDF / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


nf.save = save
if FONT:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def _letters(fig, axes, labels="abc"):
    for ax, t in zip(axes, labels):
        b = ax.get_position()
        fig.text(max(b.x0 - 0.08, 0.0), b.y1 + 0.035, t, fontsize=11, fontweight="bold", va="bottom")


def perfusion():
    """Figure 3 laid out for one page width: (a, b) on top, the real-data bars (c) underneath."""
    def layout(*a, **kw):
        fig = _figure(figsize=(6.6, 6.6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.65, wspace=0.45)
        axes = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :]))
        _letters(fig, axes)
        return fig, axes
    plt.subplots = layout
    try:
        nf.fig_perfusion()
    finally:
        plt.subplots = capped_subplots


def composite():
    """Figure 5: the three real-data panels stacked as vector subfigures (a, b, c)."""
    fig = _figure(figsize=(6.7, 9.0))
    sfs = fig.subfigures(3, 1, height_ratios=[3.0, 2.6, 3.4], hspace=0.04)
    nf.save = lambda f, name: None
    try:
        for sf, fn, t in zip(sfs, (nf.fig_q4, nf.fig_profile, nf.fig_fingerprint), "abc"):
            def sub(*a, _sf=sf, **kw):
                kw.pop("figsize", None)
                if _sf is sfs[2]:                   # fingerprint: room for the colourbar label and long ticks
                    kw["gridspec_kw"] = {"width_ratios": [1.25, 1], "wspace": 1.25, "left": 0.17, "right": 0.80}
                return _sf, _sf.subplots(*a, **kw)
            plt.subplots = sub
            fn()
            sf.text(0.0, 0.99, t, fontsize=11, fontweight="bold", va="top")
    finally:
        plt.subplots = capped_subplots
        nf.save = save
    save(fig, "real_data_composite")


MAIN = {  # journal figure -> (renderer, file stem)
    "Figure1": (nf.fig_q1, "q1_apparent_enhancement"),
    "Figure2": (nf.fig_q2, "q2_flow_speed_recovery"),
    "Figure3": (perfusion, "q8_perfusion"),
    "Figure4": (nf.fig_pipeline, "q7_published_pipeline"),
    "Figure5": (composite, "real_data_composite"),
    "Figure6": (nf.fig_q5, "q5_nuisance_fits"),
    "Figure7": (nf.fig_groups, "q9_groups"),
    "FigureS2": (nf.fig_q6, "q6_correlated_noise"),
    "FigureS3": (nf.fig_robust, "robustness_geometry_mesh"),
}


def export(stem, name):
    """Scale to at most 170 mm wide, then write EPS (vector), 600 dpi LZW TIFF and a PNG preview."""
    src = PDF / f"{stem}.pdf"
    w_pt, h_pt = [float(v) for v in subprocess.run(["pdfinfo", str(src)], capture_output=True, text=True)
                  .stdout.split("Page size:")[1].split("pts")[0].split("x")]
    f = min(1.0, MAX_MM / 25.4 * 72 / w_pt, MAX_H_MM / 25.4 * 72 / h_pt)
    fin = PDF / f"{name}.pdf"
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite", "-dFIXEDMEDIA", "-dPDFFitPage",
                    f"-dDEVICEWIDTHPOINTS={w_pt * f:.2f}", f"-dDEVICEHEIGHTPOINTS={h_pt * f:.2f}",
                    f"-sOutputFile={fin}", str(src)], check=True)
    subprocess.run(["pdftops", "-eps", str(fin), str(OUT / f"{name}.eps")], check=True)
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=tiff24nc", "-sCompression=lzw", "-r600",
                    "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", f"-sOutputFile={OUT / f'{name}.tif'}", str(fin)],
                   check=True)
    subprocess.run(["pdftocairo", "-png", "-r", "150", "-singlefile", str(fin), str(OUT / "preview" / name)],
                   check=True)
    return w_pt * f / 72 * 25.4, h_pt * f / 72 * 25.4, f


if __name__ == "__main__":
    (OUT / "preview").mkdir(parents=True, exist_ok=True)
    for name, (fn, stem) in MAIN.items():
        fn()
        w, h, f = export(stem, name.replace("Figure", PREFIX))
        print(f"{name}: {w:.0f} x {h:.0f} mm (scale {f:.2f})")
