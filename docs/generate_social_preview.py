#!/usr/bin/env python3
"""Generate GitHub social preview image for TurboQuant (1280x640)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

fig = Figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor("#0d1117")

ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_facecolor("#0d1117")
ax.axis("off")

# Title
ax.text(0.5, 0.62, "TurboQuant", fontsize=64, fontweight="bold",
        color="white", ha="center", va="center",
        fontfamily="sans-serif")

# Metrics line
ax.text(0.5, 0.44, "95.3% recall  |  5.3x compression  |  Pure Python",
        fontsize=22, color="#8b949e", ha="center", va="center",
        fontfamily="sans-serif")

# Publication line
ax.text(0.5, 0.33, "ICLR 2026  |  arXiv:2504.19874",
        fontsize=18, color="#58a6ff", ha="center", va="center",
        fontfamily="sans-serif")

# Bottom branding
ax.text(0.5, 0.10, "Firmamento Technologies",
        fontsize=14, color="#484f58", ha="center", va="center",
        fontfamily="sans-serif")

fig.savefig("/root/TurboQuant/docs/social_preview.png",
            dpi=100, facecolor="#0d1117", bbox_inches=None,
            pad_inches=0)
print("Done: social_preview.png")
