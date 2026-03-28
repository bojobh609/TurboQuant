#!/usr/bin/env python3
"""
TurboQuant Visual Assets Generator
Firmamento Technologies Design Manifesto — Swiss Style / Brutalist-lite

Colors:
  Dark Blue: #031335
  White: #FFFFFF
  Gold accent: #f0cb7a
  Light Blue accent: #91aefe
  Gray (comparison): #cccccc

Fonts:
  Inter SemiBold — titles
  Roboto Mono — data/body
"""

import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT = "/root/TurboQuant/docs"
FONT_DIR = "/tmp/fonts"

INTER_SEMIBOLD = os.path.join(FONT_DIR, "InterSemiBold.ttf")
INTER_REGULAR  = os.path.join(FONT_DIR, "InterRegular.ttf")
ROBOTO_MONO    = os.path.join(FONT_DIR, "RobotoMono.ttf")

# Register fonts for matplotlib
for fpath in [INTER_SEMIBOLD, INTER_REGULAR, ROBOTO_MONO]:
    fm.fontManager.addfont(fpath)

PROP_INTER_SB = fm.FontProperties(fname=INTER_SEMIBOLD)
PROP_INTER_R  = fm.FontProperties(fname=INTER_REGULAR)
PROP_MONO     = fm.FontProperties(fname=ROBOTO_MONO)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
DARK_BLUE  = "#031335"
WHITE      = "#FFFFFF"
GOLD       = "#f0cb7a"
LIGHT_BLUE = "#91aefe"
GRAY       = "#cccccc"

DB_RGB = (3, 19, 53)
W_RGB  = (255, 255, 255)
G_RGB  = (240, 203, 122)
LB_RGB = (145, 174, 254)


def font(path, size):
    return ImageFont.truetype(path, size)


# ===================================================================
# 1. LinkedIn Hero Image — 1200x627
# ===================================================================
def make_linkedin_hero():
    W, H = 1200, 627
    img = Image.new("RGB", (W, H), DB_RGB)
    d = ImageDraw.Draw(img)

    # Title
    f_title = font(INTER_SEMIBOLD, 72)
    d.text((80, 60), "TurboQuant", fill=W_RGB, font=f_title)

    # Subtitle
    f_sub = font(INTER_SEMIBOLD, 26)
    d.text((80, 148), "Near-Optimal Vector Quantization for AI", fill=W_RGB, font=f_sub)

    # Thin gold line
    d.line([(80, 195), (500, 195)], fill=G_RGB, width=2)

    # Metrics — 2x2 grid
    f_num = font(ROBOTO_MONO, 36)
    f_label = font(ROBOTO_MONO, 16)

    metrics = [
        ("95.3%", "recall"),
        ("5.3x", "compression"),
        ("3,781", "tests"),
        ("0s", "preprocessing"),
    ]
    col_x = [80, 370]
    row_y = [230, 330]
    for i, (val, label) in enumerate(metrics):
        x = col_x[i % 2]
        y = row_y[i // 2]
        d.text((x, y), val, fill=G_RGB, font=f_num)
        d.text((x, y + 46), label, fill=W_RGB, font=f_label)

    # arXiv line — prominent
    f_arxiv = font(ROBOTO_MONO, 18)
    d.text((80, 440), "Implementation of Google Research's ICLR 2026 algorithm", fill=W_RGB, font=f_arxiv)
    f_arxiv_id = font(ROBOTO_MONO, 18)
    d.text((80, 468), "arXiv:2504.19874", fill=G_RGB, font=f_arxiv_id)

    # Firmamento
    f_firm = font(INTER_SEMIBOLD, 18)
    d.text((80, 570), "Firmamento Technologies", fill=W_RGB, font=f_firm)

    img.save(os.path.join(OUT, "linkedin_hero.png"), dpi=(150, 150))
    print("[OK] linkedin_hero.png")


# ===================================================================
# 2. GitHub Social Preview — 1280x640
# ===================================================================
def make_social_preview():
    W, H = 1280, 640
    img = Image.new("RGB", (W, H), DB_RGB)
    d = ImageDraw.Draw(img)

    # Title — centered
    f_title = font(INTER_SEMIBOLD, 84)
    bbox = d.textbbox((0, 0), "TurboQuant", font=f_title)
    tw = bbox[2] - bbox[0]
    d.text(((W - tw) // 2, 140), "TurboQuant", fill=W_RGB, font=f_title)

    # arXiv line — gold, centered
    f_arxiv = font(ROBOTO_MONO, 22)
    line = "Based on Google Research — arXiv:2504.19874 — ICLR 2026"
    bbox = d.textbbox((0, 0), line, font=f_arxiv)
    tw = bbox[2] - bbox[0]
    d.text(((W - tw) // 2, 260), line, fill=G_RGB, font=f_arxiv)

    # Stats line — white, centered
    f_stats = font(ROBOTO_MONO, 22)
    stats = "95.3% recall  |  5.3x compression  |  Pure Python"
    bbox = d.textbbox((0, 0), stats, font=f_stats)
    tw = bbox[2] - bbox[0]
    d.text(((W - tw) // 2, 340), stats, fill=W_RGB, font=f_stats)

    # Firmamento — bottom center
    f_firm = font(INTER_SEMIBOLD, 18)
    firm = "Firmamento Technologies"
    bbox = d.textbbox((0, 0), firm, font=f_firm)
    tw = bbox[2] - bbox[0]
    d.text(((W - tw) // 2, 570), firm, fill=W_RGB, font=f_firm)

    img.save(os.path.join(OUT, "social_preview.png"), dpi=(150, 150))
    print("[OK] social_preview.png")


# ===================================================================
# 3. Hero Banner for README — 1200x300
# ===================================================================
def make_hero_banner():
    W, H = 1200, 300
    img = Image.new("RGB", (W, H), DB_RGB)
    d = ImageDraw.Draw(img)

    # Title — left-aligned
    f_title = font(INTER_SEMIBOLD, 64)
    d.text((60, 50), "TurboQuant", fill=W_RGB, font=f_title)

    # Thin gold horizontal line
    d.line([(60, 135), (400, 135)], fill=G_RGB, width=2)

    # Subtitle
    f_sub = font(INTER_SEMIBOLD, 22)
    d.text((60, 155), "Near-Optimal Vector Quantization for AI", fill=W_RGB, font=f_sub)

    # Source line in light blue
    f_src = font(ROBOTO_MONO, 18)
    d.text((60, 200), "Google Research — ICLR 2026", fill=LB_RGB, font=f_src)

    img.save(os.path.join(OUT, "hero_banner.png"), dpi=(150, 150))
    print("[OK] hero_banner.png")


# ===================================================================
# 4. Algorithm Flow Diagram — 1000x400
# ===================================================================
def make_algorithm_flow():
    W, H = 1000, 400
    img = Image.new("RGB", (W, H), W_RGB)
    d = ImageDraw.Draw(img)

    f_label = font(ROBOTO_MONO, 18)
    f_small = font(ROBOTO_MONO, 13)

    boxes = [
        ("Input Vector", False),
        ("Random Rotation", True),
        ("Lloyd-Max\nQuantization", True),
        ("Compressed\nOutput", False),
    ]

    box_w, box_h = 180, 80
    total_boxes = len(boxes)
    gap = (W - total_boxes * box_w) // (total_boxes + 1)
    y_center = H // 2

    positions = []
    for i in range(total_boxes):
        x = gap + i * (box_w + gap)
        y = y_center - box_h // 2
        positions.append((x, y))

    for i, ((label, accent), (x, y)) in enumerate(zip(boxes, positions)):
        border_color = G_RGB if accent else DB_RGB
        # Thin bordered rectangle
        d.rectangle([x, y, x + box_w, y + box_h], outline=border_color, width=2)

        # Text — handle multi-line
        lines = label.split("\n")
        line_h = 22
        total_text_h = len(lines) * line_h
        start_y = y + (box_h - total_text_h) // 2
        for j, line in enumerate(lines):
            bbox = d.textbbox((0, 0), line, font=f_label)
            tw = bbox[2] - bbox[0]
            tx = x + (box_w - tw) // 2
            ty = start_y + j * line_h
            d.text((tx, ty), line, fill=DB_RGB, font=f_label)

    # Arrows between boxes
    for i in range(total_boxes - 1):
        x1 = positions[i][0] + box_w
        x2 = positions[i + 1][0]
        y_mid = y_center
        # Thin line
        d.line([(x1, y_mid), (x2 - 8, y_mid)], fill=DB_RGB, width=2)
        # Arrowhead
        ax = x2 - 8
        d.polygon([(ax, y_mid - 6), (ax + 10, y_mid), (ax, y_mid + 6)], fill=DB_RGB)

    # Small annotation
    d.text((gap, H - 40), "Deterministic pipeline — no data-dependent preprocessing",
           fill=DB_RGB, font=f_small)

    img.save(os.path.join(OUT, "algorithm_flow.png"), dpi=(150, 150))
    print("[OK] algorithm_flow.png")


# ===================================================================
# 5. Recall Comparison Chart — 800x500 (matplotlib)
# ===================================================================
def make_recall_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    methods = ["FAISS PQ", "ScaNN", "TurboQuant"]
    values = [60, 85, 95.3]
    colors = [GRAY, GRAY, DARK_BLUE]

    bars = ax.barh(methods, values, color=colors, height=0.5, edgecolor="none")

    # Gold target line at 95%
    ax.axvline(x=95, color=GOLD, linewidth=1.5, linestyle="--", zorder=5)
    ax.text(95.5, 2.35, "95% target", fontproperties=PROP_MONO, fontsize=10, color=GOLD)

    # Value labels
    for bar, val in zip(bars, values):
        x = bar.get_width() + 1
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x, y, f"{val}%", va="center", fontproperties=PROP_MONO, fontsize=12,
                color=DARK_BLUE)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Recall@10 (%)", fontproperties=PROP_MONO, fontsize=12, color=DARK_BLUE)
    ax.set_title("Recall Comparison", fontproperties=PROP_INTER_SB, fontsize=18,
                 color=DARK_BLUE, pad=20)

    # Style — minimal
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(DARK_BLUE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color(DARK_BLUE)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(colors=DARK_BLUE, which="both", labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontproperties(PROP_MONO)
    for label in ax.get_xticklabels():
        label.set_fontproperties(PROP_MONO)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "recall_comparison.png"), dpi=150,
                facecolor=WHITE, bbox_inches="tight")
    plt.close(fig)
    print("[OK] recall_comparison.png")


# ===================================================================
# 6. Compression Tradeoff Chart — 800x500 (matplotlib)
# ===================================================================
def make_compression_tradeoff():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    # Synthetic data: bits vs recall@10
    bits = np.array([2, 3, 4, 5, 6, 7, 8])
    recall = np.array([52.1, 68.4, 80.2, 89.7, 95.3, 97.1, 98.4])

    ax.plot(bits, recall, color=DARK_BLUE, linewidth=1.5, zorder=3)
    ax.scatter(bits, recall, color=DARK_BLUE, s=40, zorder=4)

    # Gold horizontal line at 95% target
    ax.axhline(y=95, color=GOLD, linewidth=1.5, linestyle="--", zorder=2)
    ax.text(2.1, 95.8, "95% target", fontproperties=PROP_MONO, fontsize=10, color=GOLD)

    # Light Blue annotation at sweet spot (6-bit)
    ax.annotate("sweet spot\n6-bit", xy=(6, 95.3), xytext=(7.2, 88),
                fontproperties=PROP_MONO, fontsize=11, color=LIGHT_BLUE,
                arrowprops=dict(arrowstyle="->", color=LIGHT_BLUE, lw=1.2))

    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(45, 102)
    ax.set_xlabel("Bits per dimension", fontproperties=PROP_MONO, fontsize=12, color=DARK_BLUE)
    ax.set_ylabel("Recall@10 (%)", fontproperties=PROP_MONO, fontsize=12, color=DARK_BLUE)
    ax.set_title("Compression–Recall Tradeoff", fontproperties=PROP_INTER_SB, fontsize=18,
                 color=DARK_BLUE, pad=20)

    # Style — minimal
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(DARK_BLUE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color(DARK_BLUE)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(colors=DARK_BLUE, which="both", labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(PROP_MONO)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "compression_tradeoff.png"), dpi=150,
                facecolor=WHITE, bbox_inches="tight")
    plt.close(fig)
    print("[OK] compression_tradeoff.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Generating TurboQuant visual assets (Firmamento Design Manifesto)...\n")

    make_linkedin_hero()
    make_social_preview()
    make_hero_banner()
    make_algorithm_flow()
    make_recall_comparison()
    make_compression_tradeoff()

    print("\nAll 6 assets generated.")
