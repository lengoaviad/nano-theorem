"""Cover's theorem: the geometry of separability.

Cover (1965) showed that N points in general position in R^d, with random
binary labels, are linearly separable with probability
P(N, d) = (1/2^{N-1}) Σ_{k=0}^{d-1} C(N-1, k). When d ≥ N, P = 1.
This is the geometric foundation of the kernel trick: mapping to high
dimensions makes problems linearly separable.

Reference: Cover, "Geometrical and Statistical Properties of Systems of Linear
Inequalities", IEEE Trans. Electronic Computers, 1965.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.special import comb


def _cover_probability(N: int, d: int) -> float:
    """Exact Cover probability: fraction of linearly separable dichotomies.

    P(N, d) = (1/2^{N-1}) Σ_{k=0}^{d-1} C(N-1, k)

    Args:
        N: Number of points.
        d: Ambient dimension.

    Returns:
        Probability of linear separability.
    """
    if d >= N:
        return 1.0
    return sum(comb(N - 1, k, exact=True) for k in range(d)) / 2 ** (N - 1)


def _check_separability(X: np.ndarray, y: np.ndarray) -> bool:
    """Check if points X with labels y ∈ {-1, +1} are linearly separable.

    Uses linear programming: find w such that y_i (w · x_i) ≥ 1 for all i.
    Feasible iff linearly separable.

    Args:
        X: Data matrix, shape (N, d).
        y: Labels, shape (N,) with values in {-1, +1}.

    Returns:
        True if linearly separable.
    """
    N, d = X.shape
    # LP: minimize 0 subject to y_i (w · x_i) ≥ 1
    # Rewrite: -y_i (w · x_i) ≤ -1
    A_ub = -y[:, None] * X
    b_ub = -np.ones(N)
    c = np.zeros(d)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs")
    return result.success


def _empirical_separability(
    N: int, d: int, n_trials: int, rng: np.random.Generator
) -> float:
    """Estimate separability probability by Monte Carlo.

    Args:
        N: Number of points.
        d: Ambient dimension.
        n_trials: Number of random trials.
        rng: Random number generator.

    Returns:
        Fraction of trials where random dichotomy is separable.
    """
    count = 0
    for _ in range(n_trials):
        X = rng.standard_normal((N, d))
        y = rng.choice([-1, 1], size=N)
        if _check_separability(X, y):
            count += 1
    return count / n_trials


def demonstrate(
    N: int = 20,
    d_values: list[int] | None = None,
    n_trials: int = 200,
    seed: int = 42,
) -> dict:
    """Demonstrate Cover's theorem: separability phase transition.

    Args:
        N: Number of points.
        d_values: Dimensions to test. Default: range(1, 41).
        n_trials: Monte Carlo trials per (N, d) pair.
        seed: Random seed.

    Returns:
        Dict with keys: 'N', 'd_values', 'exact_probs', 'empirical_probs'.
    """
    if d_values is None:
        d_values = list(range(1, 41))

    rng = np.random.default_rng(seed)

    exact_probs = {}
    empirical_probs = {}

    for d in d_values:
        exact_probs[d] = _cover_probability(N, d)
        empirical_probs[d] = _empirical_separability(N, d, n_trials, rng)

    return {
        "N": N,
        "d_values": d_values,
        "exact_probs": exact_probs,
        "empirical_probs": empirical_probs,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
    })

    result = demonstrate(n_trials=100)  # Fewer trials for quick standalone test

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    ds = result["d_values"]
    N = result["N"]
    exact = [result["exact_probs"][d] for d in ds]
    empirical = [result["empirical_probs"][d] for d in ds]

    # Plot ratio d/N on x-axis to show universality of transition
    ratios = np.array(ds) / N
    ax.plot(ratios, exact, "-", color="#9467bd", linewidth=1.5,
            label="Cover (exact)")
    ax.scatter(ratios, empirical, s=20, color="#9467bd", alpha=0.5, zorder=5,
               label="Empirical")
    ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.3,
               label="d/N = 0.5")
    ax.set_xlabel("d / N")
    ax.set_ylabel("P(linearly separable)")
    ax.set_title(f"Cover's Theorem — Phase Transition (N = {N})",
                 fontweight="bold")
    ax.legend(fontsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("figures/cover.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"P(N={N}, d={N}) = {result['exact_probs'].get(N, 'N/A')}")
