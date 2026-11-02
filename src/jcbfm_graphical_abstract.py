"""Graphical abstract for the JCBFM submission. Usage: python jcbfm_graphical_abstract.py <outdir>"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
BLUE, RED, GREEN, GREY, PURPLE = "#4f81bd", "#c0504d", "#9bbb59", "0.35", "#8064a2"

fig = plt.figure(figsize=(7.0, 3.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=GREY))


ax.text(7.0, 6.85, "What do brain transport models identify from intrathecal tracer MRI?", ha="center",
        va="center", fontsize=10.5, fontweight="bold")

# left: schematic of tracer entering cortex from CSF
cx, cy = 1.9, 3.5
for r, c in zip(np.linspace(1.55, 0.2, 28), plt.cm.Reds(np.linspace(0.75, 0.05, 28))):
    ax.add_patch(Wedge((cx, cy), r, 0, 360, fc=c, ec="none"))
ax.add_patch(Wedge((cx, cy), 1.85, 0, 360, width=0.3, fc="#9ec5e8", ec=BLUE, lw=0.8))
ax.text(cx, cy + 2.15, "tracer in CSF", ha="center", fontsize=7.5, color=BLUE)
ax.text(cx, cy - 2.25, "MRI at 4, 24, 48, 70 h", ha="center", fontsize=7.5)
ax.text(cx, cy, "brain", ha="center", va="center", fontsize=8, color="0.25")
arrow(3.95, 3.5, 4.75, 3.5)
ax.text(4.35, 3.8, "fit", ha="center", fontsize=7.5, color=GREY)

# middle: the observed signal and its three readings
box(4.85, 4.75, 3.9, 1.35, "#f4f4f4", GREY)
ax.text(6.8, 5.62, "near-surface tracer excess", ha="center", fontsize=8.5, fontweight="bold")
ax.text(6.8, 5.1, "more tracer near the surface than plain\ndiffusion from the measured CSF predicts",
        ha="center", va="center", fontsize=6.5)
readings = [("enhanced diffusion", "fitted α > 1", RED), ("CSF perfusion of cortex", "15–45 mL/h → apparent α 1.4–10", GREEN),
            ("CSF-to-tissue calibration", "tissue gets ~0.7× the measured input", PURPLE)]
for k, (t, s, c) in enumerate(readings):
    y = 3.55 - 1.3 * k
    box(4.85, y - 0.45, 3.9, 0.95, "white", c, lw=1.3)
    ax.text(6.8, y + 0.15, t, ha="center", fontsize=8, color=c, fontweight="bold")
    ax.text(6.8, y - 0.2, s, ha="center", fontsize=6.5)
ax.text(6.8, 0.12, "indistinguishable in these data", ha="center", fontsize=7.5, style="italic")
arrow(6.8, 4.72, 6.8, 4.1)
arrow(8.85, 3.5, 9.55, 3.5)

# right: key findings
box(9.65, 0.35, 4.15, 5.75, "#fbfbfb", GREY)
ax.text(11.72, 5.68, "Known-truth tests\n3 brain geometries + human data", ha="center", va="center", fontsize=7.5, fontweight="bold")
items = [("Tissue-curve fits", "input + partial-volume errors turn\nα = 1 into α = 4.3–4.7", RED),
         ("Published pipeline", "not inflated; human data give\nα ≈ 1.3–1.5 (true 1.1–1.8)", BLUE),
         ("Uncertainty", "sampling ±0.05, analysis choices\n0.9–2.2", GREY),
         ("Group comparisons", "CSF kinetics alone: spurious\n22–33% differences in α", PURPLE)]
for k, (t, s, c) in enumerate(items):
    y = 4.75 - 1.15 * k
    ax.add_patch(plt.Circle((10.0, y + 0.02), 0.09, color=c))
    ax.text(10.2, y + 0.02, t, va="center", fontsize=7.5, fontweight="bold")
    ax.text(10.2, y - 0.42, s, va="center", fontsize=6.5)

for ext, kw in (("png", {"dpi": 300}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("pdf", {})):
    fig.savefig(OUT / f"Graphical_abstract.{ext}", **kw)
print("written", OUT)
