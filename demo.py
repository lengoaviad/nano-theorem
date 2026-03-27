"""Demonstrate six foundational approximation theorems.

Usage:
    uv run demo.py --theorem bernstein
    uv run demo.py --theorem all
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from theorems.bernstein import demonstrate as demo_bernstein
from theorems.barron import demonstrate as demo_barron
from theorems.jl import demonstrate as demo_jl
from theorems.rff import demonstrate as demo_rff
from theorems.cover import demonstrate as demo_cover
from theorems.deeponet import demonstrate as demo_deeponet

THEOREMS = ["bernstein", "barron", "jl", "rff", "cover", "deeponet"]

THEOREM_NAMES = {
    "bernstein": "Bernstein / Weierstrass (1912)",
    "barron": "Barron (1993)",
    "jl": "Johnson–Lindenstrauss (1984)",
    "rff": "Random Fourier Features (2007)",
    "cover": "Cover (1965)",
    "deeponet": "Chen & Chen / DeepONet (1995)",
}

COLORS = {
    "bernstein": "#1f77b4",
    "barron": "#ff7f0e",
    "jl": "#2ca02c",
    "rff": "#d62728",
    "cover": "#9467bd",
    "deeponet": "#8c564b",
}


def _setup_rcparams(fontsize: int = 10) -> None:
    """Set matplotlib rc parameters matching nano-solver style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": fontsize,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
    })


def _clean_axes(ax) -> None:
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_bernstein(ax, result: dict) -> str:
    """Plot Bernstein approximations on a single axis."""
    ax.plot(result["x"], result["f_true"], "k-", linewidth=1.2, label="f(x)")
    cmap = plt.cm.Blues
    for i, n in enumerate(result["degrees"]):
        c = cmap(0.3 + 0.7 * i / (len(result["degrees"]) - 1))
        ax.plot(result["x"], result["approximations"][n], "--",
                color=c, linewidth=1.0, label=f"n={n}")
    ax.set_xlabel("x")
    ax.set_title("Bernstein / Weierstrass", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, ncol=2)
    _clean_axes(ax)
    return f"max_error(n=100) = {result['max_errors'][100]:.3f}"


def plot_barron(ax, result: dict) -> str:
    """Plot Barron convergence on a single axis."""
    Ns = np.array(result["neuron_counts"], dtype=float)
    errs = [result["l2_errors"][N] for N in result["neuron_counts"]]
    C_f = result["barron_constant"]
    ax.loglog(Ns, errs, "o-", color=COLORS["barron"], linewidth=1.2,
              markersize=4, label="L2 error")
    ax.loglog(Ns, C_f / np.sqrt(Ns), "k--", linewidth=0.8, alpha=0.5,
              label=r"$C_f/\sqrt{N}$")
    ax.set_xlabel("Neurons N")
    ax.set_ylabel("L2 Error")
    ax.set_title("Barron (1993)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6)
    _clean_axes(ax)
    N_last = result["neuron_counts"][-1]
    return f"L2_error(N={N_last}) = {result['l2_errors'][N_last]:.3f}"


def plot_jl(ax, result: dict) -> str:
    """Plot JL distortion vs. target dimension."""
    ks = result["d_targets"]
    dists = [result["max_distortions"][k] for k in ks]
    eps = result["epsilon"]
    ax.semilogx(ks, dists, "o-", color=COLORS["jl"], linewidth=1.2, markersize=4)
    ax.axhline(eps, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"ε = {eps}")
    ax.set_xlabel("Target dim k")
    ax.set_ylabel("Max distortion")
    ax.set_title("Johnson–Lindenstrauss (1984)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6)
    _clean_axes(ax)
    return f"k* = {result['theoretical_k']}"


def plot_rff(ax, result: dict) -> str:
    """Plot RFF kernel approximation error."""
    Ds = result["D_values"]
    kerr = [result["kernel_errors"][D] for D in Ds]
    ax.loglog(Ds, kerr, "o-", color=COLORS["rff"], linewidth=1.2, markersize=4)
    ax.set_xlabel("Features D")
    ax.set_ylabel("Rel. kernel error")
    ax.set_title("Random Fourier Features (2007)", fontsize=9, fontweight="bold")
    _clean_axes(ax)
    D_last = Ds[-1]
    return f"kernel_err(D={D_last}) = {result['kernel_errors'][D_last]:.4f}"


def plot_cover(ax, result: dict) -> str:
    """Plot Cover separability phase transition."""
    ds = result["d_values"]
    N = result["N"]
    exact = [result["exact_probs"][d] for d in ds]
    empirical = [result["empirical_probs"][d] for d in ds]
    ratios = np.array(ds) / N
    ax.plot(ratios, exact, "-", color=COLORS["cover"], linewidth=1.2,
            label="Exact")
    ax.scatter(ratios, empirical, s=12, color=COLORS["cover"], alpha=0.4,
               zorder=5, label="Empirical")
    ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("d / N")
    ax.set_ylabel("P(separable)")
    ax.set_title("Cover (1965)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6)
    _clean_axes(ax)
    d_half = N // 2
    return f"P(N={N}, d={d_half}) = {result['exact_probs'].get(d_half, 0.0):.3f}"


def plot_deeponet(ax, result: dict) -> str:
    """Plot DeepONet example predictions."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (ex, c) in enumerate(zip(result["example_data"], colors)):
        label_t = f"True #{i+1}" if i == 0 else None
        ax.plot(ex["y_fine"], ex["Gu_true"], "-", color=c, linewidth=1.2,
                label=label_t)
        ax.plot(ex["y_fine"], ex["Gu_pred"], "--", color=c, linewidth=1.0)
    ax.set_xlabel("y")
    ax.set_ylabel("G(u)(y)")
    ax.set_title("DeepONet (1995)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, labels=["True", "Predicted"])
    _clean_axes(ax)
    return f"test_RMSE = {result['test_error']:.4f}"


def plot_single_bernstein(result: dict, save_path: str) -> None:
    """Detailed single-theorem plot for bernstein."""
    _setup_rcparams()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(result["x"], result["f_true"], "k-", linewidth=1.5, label="f(x)")
    cmap = plt.cm.Blues
    for i, n in enumerate(result["degrees"]):
        c = cmap(0.3 + 0.7 * i / (len(result["degrees"]) - 1))
        ax1.plot(result["x"], result["approximations"][n], "--",
                 color=c, linewidth=1.2, label=f"n = {n}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("Bernstein Polynomial Approximation", fontweight="bold")
    ax1.legend(fontsize=8)
    _clean_axes(ax1)

    ns = result["degrees"]
    errs = [result["max_errors"][n] for n in ns]
    ax2.loglog(ns, errs, "o-", color=COLORS["bernstein"], linewidth=1.5,
               markersize=5)
    ax2.loglog(ns, errs[0] * np.sqrt(ns[0] / np.array(ns, dtype=float)),
               "k--", linewidth=1.0, alpha=0.5, label=r"$O(1/\sqrt{n})$")
    ax2.set_xlabel("Degree n")
    ax2.set_ylabel("Max Error")
    ax2.set_title("Convergence Rate", fontweight="bold")
    ax2.legend(fontsize=8)
    _clean_axes(ax2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_single_generic(result: dict, theorem: str, save_path: str) -> None:
    """Detailed single-theorem plot (two panels) for non-bernstein theorems."""
    _setup_rcparams()
    plot_fns = {
        "barron": plot_barron,
        "jl": plot_jl,
        "rff": plot_rff,
        "cover": plot_cover,
        "deeponet": plot_deeponet,
    }
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    plot_fns[theorem](ax, result)
    ax.set_title(THEOREM_NAMES[theorem], fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate foundational approximation theorems."
    )
    parser.add_argument(
        "--theorem", default="bernstein",
        choices=THEOREMS + ["all"],
        help="Which theorem to demonstrate (default: bernstein)",
    )
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    theorems = THEOREMS if args.theorem == "all" else [args.theorem]

    # Run all demonstrations
    all_results = {}
    for theorem in theorems:
        start = time.time()
        print(f"{THEOREM_NAMES[theorem]:<40}", end="", flush=True)

        if theorem == "bernstein":
            result = demo_bernstein()
        elif theorem == "barron":
            result = demo_barron()
        elif theorem == "jl":
            result = demo_jl()
        elif theorem == "rff":
            result = demo_rff()
        elif theorem == "cover":
            result = demo_cover(n_trials=100)
        elif theorem == "deeponet":
            result = demo_deeponet(epochs=2000)

        elapsed = time.time() - start
        all_results[theorem] = result

        # Print summary metric
        if theorem == "bernstein":
            metric = f"max_error(n=100) = {result['max_errors'][100]:.3f}"
        elif theorem == "barron":
            N_last = result["neuron_counts"][-1]
            metric = f"L2_error(N={N_last}) = {result['l2_errors'][N_last]:.3f}"
        elif theorem == "jl":
            metric = f"k* = {result['theoretical_k']}"
        elif theorem == "rff":
            metric = f"kernel_err(D=500) = {result['kernel_errors'][500]:.4f}"
        elif theorem == "cover":
            N = result["N"]
            d_half = N // 2
            metric = f"P(N={N}, d={d_half}) = {result['exact_probs'].get(d_half, 0.0):.3f}"
        elif theorem == "deeponet":
            metric = f"test_RMSE = {result['test_error']:.4f}"
        print(f"{elapsed:>6.2f}s  {metric}")

    # Plot
    if args.theorem == "all":
        _setup_rcparams(fontsize=9)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes_flat = axes.flatten()

        plot_fns = [
            plot_bernstein, plot_barron, plot_jl,
            plot_rff, plot_cover, plot_deeponet,
        ]
        for ax, theorem, plot_fn in zip(axes_flat, THEOREMS, plot_fns):
            plot_fn(ax, all_results[theorem])

        fig.suptitle("Six Foundational Approximation Theorems",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig("figures/all_theorems.png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        print(f"\nSaved figures/all_theorems.png")
    else:
        theorem = args.theorem
        result = all_results[theorem]
        save_path = f"figures/{theorem}.png"
        if theorem == "bernstein":
            plot_single_bernstein(result, save_path)
        else:
            plot_single_generic(result, theorem, save_path)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
