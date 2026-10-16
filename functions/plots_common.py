"""Shared matplotlib style."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(blue="#1f6fb4", orange="#e8710a", green="#2a9d4b", red="#d1343a",
         purple="#7b4fa3", gray="#6b7280", teal="#16888f", dark="#1f2937")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.constrained_layout.use": True,
    "axes.titleweight": "bold",
})


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
