"""Generate figures for TurboQuant README and documentation."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# Style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#0d1117',
    'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 13,
})

COLORS = {
    'tq_main': '#58a6ff',
    'tq_accent': '#3fb950',
    'faiss': '#f0883e',
    'baseline': '#8b949e',
    'warn': '#d29922',
    'bg_card': '#161b22',
}


def fig1_recall_comparison():
    """Bar chart: TurboQuant recall vs FAISS PQ at different bit-widths."""
    fig, ax = plt.subplots(figsize=(12, 6))

    bits = ['2-bit', '3-bit', '4-bit', '5-bit', '6-bit']
    tq_recall = [59.2, 77.6, 86.2, 92.6, 95.3]
    # PQ approximate recalls for comparison (from paper benchmarks)
    pq_recall = [25, 42, 58, 68, 75]

    x = np.arange(len(bits))
    width = 0.35

    bars1 = ax.bar(x - width/2, tq_recall, width, label='TurboQuant',
                   color=COLORS['tq_main'], edgecolor='none', alpha=0.9, zorder=3)
    bars2 = ax.bar(x + width/2, pq_recall, width, label='Product Quantization',
                   color=COLORS['faiss'], edgecolor='none', alpha=0.7, zorder=3)

    # Add value labels
    for bar, val in zip(bars1, tq_recall):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val}%', ha='center', va='bottom', fontweight='bold',
                color=COLORS['tq_main'], fontsize=12)
    for bar, val in zip(bars2, pq_recall):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val}%', ha='center', va='bottom',
                color=COLORS['faiss'], fontsize=11)

    ax.set_xlabel('Quantization Bit-Width', fontsize=14)
    ax.set_ylabel('Recall@10 (%)', fontsize=14)
    ax.set_title('TurboQuant vs Product Quantization — Recall@10', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(bits, fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.3)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.axhline(y=95, color=COLORS['tq_accent'], linestyle='--', alpha=0.5, label='95% target')
    ax.text(4.5, 96, '95% target', color=COLORS['tq_accent'], fontsize=10, alpha=0.7)

    plt.tight_layout()
    fig.savefig('/root/TurboQuant/docs/recall_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  recall_comparison.png")


def fig2_compression_tradeoff():
    """Scatter plot: recall vs compression ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # TurboQuant points
    tq_bits = [2, 3, 4, 5, 6]
    tq_recall = [59.2, 77.6, 86.2, 92.6, 95.3]
    tq_compress = [16.0, 10.7, 8.0, 6.4, 5.3]

    ax.scatter(tq_compress, tq_recall, s=200, c=COLORS['tq_main'], zorder=5, edgecolors='white', linewidth=1.5)
    for b, r, c in zip(tq_bits, tq_recall, tq_compress):
        ax.annotate(f'{b}-bit', (c, r), textcoords="offset points", xytext=(12, 5),
                    fontsize=11, color=COLORS['tq_main'], fontweight='bold')

    # Connect with line
    ax.plot(tq_compress, tq_recall, color=COLORS['tq_main'], alpha=0.4, linewidth=2, zorder=4)

    # Reference: FAISS Flat (no compression)
    ax.scatter([1.0], [100], s=150, c=COLORS['baseline'], marker='D', zorder=5)
    ax.annotate('FAISS Flat\n(exact)', (1.0, 100), textcoords="offset points",
                xytext=(-60, -20), fontsize=10, color=COLORS['baseline'])

    # Reference: PQ at 8x
    ax.scatter([8.0], [58], s=120, c=COLORS['faiss'], marker='s', zorder=5)
    ax.annotate('FAISS PQ\n4-bit', (8.0, 58), textcoords="offset points",
                xytext=(15, -15), fontsize=10, color=COLORS['faiss'])

    ax.set_xlabel('Compression Ratio (x)', fontsize=14)
    ax.set_ylabel('Recall@10 (%)', fontsize=14)
    ax.set_title('Quality vs Compression Trade-off', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(0, 18)
    ax.set_ylim(40, 105)
    ax.grid(alpha=0.3, zorder=0)

    # Highlight sweet spot
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((5.85, 93.95), 2.5, 8, fill=True, facecolor=COLORS['tq_accent'],
                      alpha=0.1, edgecolor=COLORS['tq_accent'], linewidth=2, linestyle='--')
    ax.add_patch(ellipse)
    ax.text(5.85, 99, 'Sweet Spot', ha='center', fontsize=11, color=COLORS['tq_accent'], fontstyle='italic')

    plt.tight_layout()
    fig.savefig('/root/TurboQuant/docs/compression_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  compression_tradeoff.png")


def fig3_hero_banner():
    """Hero banner image for the top of README."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Background gradient effect
    for i in range(100):
        alpha = 0.02
        ax.axhline(y=i*0.05, color=COLORS['tq_main'], alpha=alpha, linewidth=0.5)

    # Title
    ax.text(7, 3.5, 'TurboQuant', fontsize=48, fontweight='bold',
            ha='center', va='center', color='white',
            path_effects=[pe.withStroke(linewidth=3, foreground=COLORS['tq_main'])])

    # Subtitle
    ax.text(7, 2.5, 'Near-Optimal Vector Quantization for AI',
            fontsize=18, ha='center', va='center', color=COLORS['baseline'])

    # Key stats
    stats = [
        ('5x', 'Compression'),
        ('95%+', 'Recall'),
        ('0ms', 'Preprocessing'),
        ('2.7x', 'Shannon Limit'),
    ]
    for i, (val, label) in enumerate(stats):
        x = 2 + i * 3.3
        ax.text(x, 1.3, val, fontsize=28, fontweight='bold',
                ha='center', color=COLORS['tq_main'])
        ax.text(x, 0.7, label, fontsize=12, ha='center', color=COLORS['baseline'])

    plt.tight_layout()
    fig.savefig('/root/TurboQuant/docs/hero_banner.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  hero_banner.png")


def fig4_algorithm_diagram():
    """Visual diagram of the TurboQuant algorithm flow."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    boxes = [
        (1.5, 2, 'Input Vector\nx (d dims)', COLORS['baseline']),
        (4.5, 2, 'Random\nRotation (QR)', COLORS['tq_main']),
        (7.5, 2, 'Lloyd-Max\nQuantize', COLORS['tq_accent']),
        (10.5, 2, 'Compressed\nb bits/coord', COLORS['warn']),
    ]

    for x, y, text, color in boxes:
        rect = FancyBboxPatch((x-1.1, y-0.7), 2.2, 1.4,
                              boxstyle="round,pad=0.15",
                              facecolor=color, alpha=0.2,
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)

    # Arrows
    for x1, x2 in [(2.6, 3.4), (5.6, 6.4), (8.6, 9.4)]:
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))

    # Labels
    ax.text(3, 2.9, 'y = Π·x', fontsize=11, ha='center', color=COLORS['tq_main'], fontstyle='italic')
    ax.text(6, 2.9, 'Beta dist.', fontsize=11, ha='center', color=COLORS['tq_accent'], fontstyle='italic')
    ax.text(9, 2.9, '8x smaller', fontsize=11, ha='center', color=COLORS['warn'], fontstyle='italic')

    ax.text(7, 0.5, 'TurboQuant Algorithm — Zero preprocessing, near-Shannon-limit compression',
            fontsize=13, ha='center', color=COLORS['baseline'], fontstyle='italic')

    plt.tight_layout()
    fig.savefig('/root/TurboQuant/docs/algorithm_flow.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  algorithm_flow.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_recall_comparison()
    fig2_compression_tradeoff()
    fig3_hero_banner()
    fig4_algorithm_diagram()
    print("Done!")
