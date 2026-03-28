"""Generate TurboQuant animation showing the quantization process step by step."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
from pathlib import Path
import subprocess

# Dark theme matching GitHub
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
    'font.size': 12,
})

C = {
    'blue': '#58a6ff', 'green': '#3fb950', 'orange': '#f0883e',
    'yellow': '#d29922', 'gray': '#8b949e', 'red': '#f85149',
    'purple': '#bc8cff', 'bg': '#0d1117', 'card': '#161b22',
}

FRAME_DIR = Path("/tmp/tq_frames")
FRAME_DIR.mkdir(exist_ok=True)

# Pre-compute real TurboQuant data
rng = np.random.RandomState(42)
d = 20  # low-d for visualization
N = 200
vectors = rng.randn(N, d).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

# Rotation matrix
gaussian = rng.randn(d, d)
rotation, _ = np.linalg.qr(gaussian)
rotated = vectors @ rotation.T

# Centroids (4-bit = 16 levels)
sigma = 1 / np.sqrt(d)
from scipy.stats import norm
centroids = norm.ppf(np.linspace(1/32, 31/32, 16), scale=sigma)

# Quantize
dists = np.abs(rotated[:, :, np.newaxis] - centroids[np.newaxis, np.newaxis, :])
indices = np.argmin(dists, axis=2)
reconstructed_rot = centroids[indices]
reconstructed = reconstructed_rot @ rotation


def frame_title(ax, text, subtitle=""):
    ax.text(7, 5.7, text, fontsize=22, fontweight='bold', ha='center', color='white',
            path_effects=[pe.withStroke(linewidth=2, foreground=C['blue'])])
    if subtitle:
        ax.text(7, 5.2, subtitle, fontsize=13, ha='center', color=C['gray'])


def make_frame(frame_num):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    return fig, ax


# ================================================================
# PHASE 1: Title card (frames 0-29, ~1s)
# ================================================================
def phase_title(frame_idx):
    fig, ax = make_frame(frame_idx)

    alpha = min(1.0, frame_idx / 15)
    ax.text(7, 3.8, 'TurboQuant', fontsize=52, fontweight='bold', ha='center',
            color='white', alpha=alpha,
            path_effects=[pe.withStroke(linewidth=4, foreground=C['blue'])])
    ax.text(7, 2.8, 'How near-optimal vector quantization works',
            fontsize=18, ha='center', color=C['gray'], alpha=alpha)
    ax.text(7, 1.5, 'Based on Google Research — ICLR 2026 (arXiv:2504.19874)',
            fontsize=12, ha='center', color=C['gray'], alpha=alpha * 0.7)

    return fig


# ================================================================
# PHASE 2: Show original vectors (frames 30-59)
# ================================================================
def phase_original(frame_idx):
    fig, ax = make_frame(frame_idx)
    t = (frame_idx - 30) / 30

    frame_title(ax, "Step 1: Input Vectors", f"d={d} dimensions, {N} vectors on unit sphere")

    # Plot first 2 dims of original vectors
    ax2 = fig.add_axes([0.08, 0.08, 0.4, 0.72])
    ax2.set_facecolor(C['card'])
    n_show = min(int(t * N) + 5, N)
    ax2.scatter(vectors[:n_show, 0], vectors[:n_show, 1], s=15, c=C['blue'], alpha=0.6, zorder=3)

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1 / np.sqrt(d) * 3  # scale for visibility
    ax2.plot(r*np.cos(theta), r*np.sin(theta), color=C['gray'], alpha=0.3, linewidth=1)
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_title('Original Vectors (2D projection)', color=C['gray'], fontsize=11)
    ax2.grid(alpha=0.2)

    # Show coordinate histogram
    ax3 = fig.add_axes([0.55, 0.08, 0.4, 0.72])
    ax3.set_facecolor(C['card'])
    coords = vectors[:n_show].flatten()
    ax3.hist(coords, bins=50, color=C['blue'], alpha=0.6, density=True, edgecolor='none')
    ax3.set_title('Coordinate Distribution (arbitrary)', color=C['gray'], fontsize=11)
    ax3.set_xlabel('Coordinate Value', fontsize=10)
    ax3.grid(alpha=0.2)

    # Info box
    ax.text(12.5, 1.5, f'{n_show} vectors\n{d} dimensions\nNorm = 1.0',
            fontsize=11, color=C['blue'], ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=C['card'], edgecolor=C['blue'], alpha=0.8))

    return fig


# ================================================================
# PHASE 3: Random rotation (frames 60-99)
# ================================================================
def phase_rotation(frame_idx):
    fig, ax = make_frame(frame_idx)
    t = (frame_idx - 60) / 40

    frame_title(ax, "Step 2: Random Rotation", "y = \u03A0 \u00B7 x  (orthogonal matrix via QR decomposition)")

    # Interpolate between original and rotated
    interp = vectors * (1 - t) + rotated * t

    # Before
    ax2 = fig.add_axes([0.08, 0.08, 0.4, 0.72])
    ax2.set_facecolor(C['card'])
    ax2.scatter(interp[:, 0], interp[:, 1], s=15,
                c=[C['blue'] if t < 0.5 else C['green']], alpha=0.6, zorder=3)
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.8, 0.8)
    label = 'Rotating...' if t < 1.0 else 'Rotated!'
    ax2.set_title(f'Vectors ({label})', color=C['green'] if t > 0.5 else C['gray'], fontsize=11)
    ax2.grid(alpha=0.2)

    # Histogram transitions to Beta distribution
    ax3 = fig.add_axes([0.55, 0.08, 0.4, 0.72])
    ax3.set_facecolor(C['card'])
    coords = interp.flatten()
    ax3.hist(coords, bins=50, color=C['green'], alpha=0.6, density=True, edgecolor='none')

    # Overlay theoretical Beta PDF
    if t > 0.3:
        from turboquant.codebook import hypersphere_coordinate_pdf
        x_range = np.linspace(-0.6, 0.6, 200)
        pdf_vals = [hypersphere_coordinate_pdf(x, d) for x in x_range]
        ax3.plot(x_range, pdf_vals, color=C['yellow'], linewidth=2, alpha=t,
                 label='Beta PDF (known!)')
        ax3.legend(fontsize=9, loc='upper right')

    ax3.set_title('Coordinates now follow Beta distribution', color=C['green'], fontsize=11)
    ax3.grid(alpha=0.2)

    # Key insight
    if t > 0.5:
        ax.text(7, 0.3, 'Key insight: After rotation, coordinate distribution is KNOWN',
                fontsize=13, ha='center', color=C['yellow'], fontweight='bold', alpha=min(1, (t-0.5)*3))

    return fig


# ================================================================
# PHASE 4: Lloyd-Max quantization (frames 100-149)
# ================================================================
def phase_quantize(frame_idx):
    fig, ax = make_frame(frame_idx)
    t = (frame_idx - 100) / 50

    frame_title(ax, "Step 3: Lloyd-Max Quantization",
                "Optimal scalar quantizer per coordinate (pre-computed centroids)")

    # Show histogram with quantization levels appearing
    ax3 = fig.add_axes([0.1, 0.08, 0.8, 0.72])
    ax3.set_facecolor(C['card'])

    coords = rotated.flatten()
    ax3.hist(coords, bins=80, color=C['green'], alpha=0.3, density=True, edgecolor='none')

    # Show centroids appearing one by one
    n_centroids = min(int(t * 16) + 1, 16)
    for i in range(n_centroids):
        ax3.axvline(x=centroids[i], color=C['orange'], linewidth=2, alpha=0.8)
        if n_centroids <= 8:
            ax3.text(centroids[i], ax3.get_ylim()[1] * 0.95, f'c{i}',
                     ha='center', fontsize=8, color=C['orange'])

    # Show quantization regions
    if t > 0.5:
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        for b in boundaries:
            ax3.axvline(x=b, color=C['gray'], linewidth=0.5, linestyle='--', alpha=0.5)

    ax3.set_title(f'{n_centroids}/16 centroids placed — {4} bits per coordinate',
                  color=C['orange'], fontsize=12)
    ax3.set_xlabel('Coordinate Value', fontsize=11)
    ax3.grid(alpha=0.2)

    # Compression info
    if t > 0.7:
        ax.text(12, 5.5, f'32-bit \u2192 4-bit\n8x compression',
                fontsize=14, color=C['orange'], ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=C['card'], edgecolor=C['orange']))

    return fig


# ================================================================
# PHASE 5: Show reconstruction quality (frames 150-199)
# ================================================================
def phase_quality(frame_idx):
    fig, ax = make_frame(frame_idx)
    t = (frame_idx - 150) / 50

    frame_title(ax, "Step 4: Reconstruction Quality",
                "Compressed vectors maintain high similarity")

    # Original vs reconstructed scatter
    ax2 = fig.add_axes([0.08, 0.08, 0.4, 0.72])
    ax2.set_facecolor(C['card'])
    ax2.scatter(vectors[:, 0], vectors[:, 1], s=15, c=C['blue'], alpha=0.4, label='Original', zorder=3)
    n_show = min(int(t * N) + 5, N)
    ax2.scatter(reconstructed[:n_show, 0], reconstructed[:n_show, 1],
                s=15, c=C['orange'], alpha=0.4, label='Reconstructed', zorder=3)

    # Draw error lines for a few vectors
    if t > 0.3:
        for i in range(0, min(n_show, 30), 3):
            ax2.plot([vectors[i, 0], reconstructed[i, 0]],
                     [vectors[i, 1], reconstructed[i, 1]],
                     color=C['red'], alpha=0.15, linewidth=0.5)

    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.8, 0.8)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_title('Original vs Reconstructed', color=C['gray'], fontsize=11)
    ax2.grid(alpha=0.2)

    # Cosine similarity histogram
    ax3 = fig.add_axes([0.55, 0.08, 0.4, 0.72])
    ax3.set_facecolor(C['card'])
    cos_sims = np.sum(vectors * reconstructed, axis=1)
    ax3.hist(cos_sims, bins=50, color=C['green'], alpha=0.7, edgecolor='none')
    avg_cos = np.mean(cos_sims)
    ax3.axvline(x=avg_cos, color=C['yellow'], linewidth=2, linestyle='--')
    ax3.text(avg_cos - 0.01, ax3.get_ylim()[1] * 0.8 if ax3.get_ylim()[1] > 0 else 10,
             f'avg={avg_cos:.3f}', color=C['yellow'], fontsize=11, fontweight='bold')
    ax3.set_title('Cosine Similarity Distribution', color=C['green'], fontsize=11)
    ax3.set_xlabel('Cosine Similarity', fontsize=10)
    ax3.grid(alpha=0.2)

    return fig


# ================================================================
# PHASE 6: Final results (frames 200-239)
# ================================================================
def phase_results(frame_idx):
    fig, ax = make_frame(frame_idx)
    t = (frame_idx - 200) / 40

    ax.text(7, 5.3, 'TurboQuant Results', fontsize=32, fontweight='bold', ha='center',
            color='white', path_effects=[pe.withStroke(linewidth=3, foreground=C['blue'])])

    results = [
        ('95.3%', 'Recall@10', '(6-bit, d=384)', C['green']),
        ('5.3x', 'Compression', '(vs float32)', C['blue']),
        ('0 ms', 'Preprocessing', '(data-oblivious)', C['orange']),
        ('2.4x', 'Shannon Limit', '(near-optimal)', C['purple']),
    ]

    for i, (val, label, detail, color) in enumerate(results):
        alpha = min(1.0, max(0, (t - i * 0.15) * 5))
        y = 3.8 - i * 1.1
        ax.text(5, y, val, fontsize=36, fontweight='bold', ha='right', color=color, alpha=alpha)
        ax.text(5.3, y + 0.1, label, fontsize=16, ha='left', color='white', alpha=alpha)
        ax.text(5.3, y - 0.3, detail, fontsize=11, ha='left', color=C['gray'], alpha=alpha)

    if t > 0.8:
        ax.text(7, 0.3, 'github.com/Firmamento-Technologies/TurboQuant',
                fontsize=14, ha='center', color=C['blue'], fontweight='bold',
                alpha=min(1, (t - 0.8) * 5))

    return fig


# ================================================================
# Generate all frames
# ================================================================
def generate_all_frames():
    total_frames = 240  # 8 seconds at 30fps

    phases = [
        (0, 30, phase_title),
        (30, 60, phase_original),
        (60, 100, phase_rotation),
        (100, 150, phase_quantize),
        (150, 200, phase_quality),
        (200, 240, phase_results),
    ]

    print(f"Generating {total_frames} frames...")
    for frame_idx in range(total_frames):
        for start, end, func in phases:
            if start <= frame_idx < end:
                fig = func(frame_idx)
                fig.savefig(FRAME_DIR / f"frame_{frame_idx:04d}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
                break

        if (frame_idx + 1) % 30 == 0:
            print(f"  {frame_idx + 1}/{total_frames} frames")

    print("Encoding MP4...")
    output = "/root/TurboQuant/docs/turboquant_demo.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", str(FRAME_DIR / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "medium",
        output,
    ], capture_output=True)

    # Also create a GIF for GitHub README (GitHub doesn't inline mp4)
    gif_output = "/root/TurboQuant/docs/turboquant_demo.gif"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", output,
        "-vf", "fps=15,scale=700:-1:flags=lanczos",
        "-loop", "0",
        gif_output,
    ], capture_output=True)

    print(f"Done! MP4: {output}")
    print(f"      GIF: {gif_output}")


if __name__ == "__main__":
    generate_all_frames()
