"""Generate hero and design-space figures for README.

Usage:
    uv run generate_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from theorems.bernstein import bernstein_approximate


def _setup_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
    })


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_hero(save_path: str = "figures/hero.png") -> None:
    """Three-panel hero: target → arrow → Bernstein approximations."""
    _setup_rcparams()

    f = lambda x: np.sin(5 * np.pi * x) * np.exp(-2 * x) + 0.5
    x = np.linspace(0, 1, 500)
    f_true = f(x)

    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 5, width_ratios=[2, 0.8, 0.4, 0.8, 2],
                          wspace=0.05)

    # Left panel: target function
    ax_left = fig.add_subplot(gs[0])
    ax_left.plot(x, f_true, "k-", linewidth=2.0)
    ax_left.fill_between(x, f_true, alpha=0.15, color="#1f77b4")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("f(x)")
    ax_left.set_title("Target Function", fontweight="bold")
    _clean_axes(ax_left)

    # Center: arrow with text
    ax_arrow = fig.add_subplot(gs[2])
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.annotate(
        "", xy=(0.95, 0.5), xytext=(0.05, 0.5),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="k"),
    )
    ax_arrow.text(0.5, 0.65, "Bernstein\npolynomials",
                  ha="center", va="bottom", fontsize=9, fontstyle="italic")
    ax_arrow.text(0.5, 0.35, "n = 5, 20, 100",
                  ha="center", va="top", fontsize=8, color="gray")
    ax_arrow.axis("off")

    # Right panel: Bernstein approximations
    ax_right = fig.add_subplot(gs[4])
    ax_right.plot(x, f_true, "k-", linewidth=1.5, alpha=0.3, label="f(x)")
    degrees = [5, 20, 100]
    blues = ["#aec7e8", "#5b9bd5", "#1f4e79"]
    for n, color in zip(degrees, blues):
        approx = bernstein_approximate(f, x, n)
        ax_right.plot(x, approx, "-", color=color, linewidth=1.5,
                      label=f"n = {n}")
    ax_right.set_xlabel("x")
    ax_right.set_title("Bernstein Approximations", fontweight="bold")
    ax_right.legend(fontsize=8)
    _clean_axes(ax_right)

    # Hide spacer axes
    for idx in [1, 3]:
        ax_spacer = fig.add_subplot(gs[idx])
        ax_spacer.axis("off")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {save_path}")


def generate_design_space(save_path: str = "figures/design_space.png") -> None:
    """2D diagram: what is approximated vs. construction type."""
    _setup_rcparams()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Axis labels
    ax.set_xlabel("What is Approximated →", fontsize=11, fontweight="bold")
    ax.set_ylabel("Construction Type →", fontsize=11, fontweight="bold")

    # Positions: (x, y) in normalized coordinates
    # x: functions (0.15) → kernels (0.5) → operators (0.85)
    # y: deterministic (0.2) → probabilistic (0.55) → learned (0.85)
    bw, bh = 0.10, 0.10  # box width, height
    theorems = {
        "Bernstein\n(1912)": (0.12, 0.2, "#1f77b4"),
        "Barron\n(1993)": (0.28, 0.55, "#ff7f0e"),
        "JL\n(1984)": (0.44, 0.55, "#2ca02c"),
        "RFF\n(2007)": (0.58, 0.55, "#d62728"),
        "Cover\n(1965)": (0.74, 0.55, "#9467bd"),
        "DeepONet\n(1995)": (0.88, 0.85, "#8c564b"),
    }

    for label, (x, y, color) in theorems.items():
        box = FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.01",
            facecolor="white", edgecolor=color, linewidth=2.0,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color=color)

    # Axis tick labels
    ax.set_xticks([0.12, 0.50, 0.88])
    ax.set_xticklabels(["Functions", "Kernels /\nDistances", "Operators"],
                       fontsize=9)
    ax.set_yticks([0.2, 0.55, 0.85])
    ax.set_yticklabels(["Deterministic", "Probabilistic", "Learned"],
                       fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("The Design Space of Approximation Theorems",
                 fontsize=12, fontweight="bold", pad=15)
    ax.grid(False)
    _clean_axes(ax)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    generate_hero()
    generate_design_space()
