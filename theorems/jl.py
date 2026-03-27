"""Johnson-Lindenstrauss lemma: random projections preserve geometry.

For any ε ∈ (0,1) and n points in R^d, a random linear map R^d → R^k with
k = O(log n / ε²) preserves all pairwise distances within factor (1 ± ε).
The construction is trivial: multiply by a Gaussian random matrix scaled by 1/√k.

Reference: Johnson & Lindenstrauss, "Extensions of Lipschitz mappings into a
Hilbert space", Contemp. Math., 1984.
"""

import numpy as np


def _random_projection(
    X: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Project X from R^d to R^k via scaled Gaussian random matrix.

    This IS the JL construction: R = randn(d, k) / √k.

    Args:
        X: Data matrix, shape (n, d).
        k: Target dimension.
        rng: Random number generator.

    Returns:
        Projected data, shape (n, k).
    """
    d = X.shape[1]
    R = rng.standard_normal((d, k)) / np.sqrt(k)
    return X @ R


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute all pairwise L2 distances (upper triangle, flattened).

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        Pairwise distances, shape (n*(n-1)/2,).
    """
    # Efficient computation: ‖xi - xj‖² = ‖xi‖² + ‖xj‖² - 2 xi·xj
    norms_sq = np.sum(X**2, axis=1)
    dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * X @ X.T

    # Extract upper triangle
    n = X.shape[0]
    idx = np.triu_indices(n, k=1)
    return np.sqrt(np.maximum(dists_sq[idx], 0))


def demonstrate(
    n_points: int = 500,
    d_original: int = 1000,
    d_targets: list[int] | None = None,
    epsilon: float = 0.3,
    seed: int = 42,
) -> dict:
    """Demonstrate the JL lemma with random Gaussian projections.

    Args:
        n_points: Number of random points.
        d_original: Original dimension.
        d_targets: Target dimensions to test. Default: [2, 5, 10, 25, 50, 100, 200, 500].
        epsilon: Distortion tolerance for theoretical bound.
        seed: Random seed.

    Returns:
        Dict with keys: 'n_points', 'd_original', 'd_targets', 'epsilon',
        'max_distortions', 'distance_ratios' (for largest k),
        'theoretical_k'.
    """
    if d_targets is None:
        d_targets = [2, 5, 10, 25, 50, 100, 200, 500]

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_points, d_original))

    # Original pairwise distances
    dist_orig = _pairwise_distances(X)

    max_distortions = {}
    ratios_at_best = None

    for k in d_targets:
        X_proj = _random_projection(X, k, rng)
        dist_proj = _pairwise_distances(X_proj)

        # Distance ratios: projected / original
        ratios = dist_proj / dist_orig
        max_distortions[k] = np.max(np.abs(ratios - 1.0))

        # Store ratios for the largest target dimension
        if k == max(d_targets):
            ratios_at_best = ratios

    # Theoretical minimum k for (1±ε) guarantee
    # k ≥ 8 ln(n) / ε² (from the standard proof)
    theoretical_k = int(np.ceil(8 * np.log(n_points) / epsilon**2))

    return {
        "n_points": n_points,
        "d_original": d_original,
        "d_targets": d_targets,
        "epsilon": epsilon,
        "max_distortions": max_distortions,
        "distance_ratios": ratios_at_best,
        "theoretical_k": theoretical_k,
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

    result = demonstrate()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: distance ratio histogram
    ax1.hist(result["distance_ratios"], bins=80, color="#2ca02c", alpha=0.7,
             edgecolor="white", linewidth=0.5, density=True)
    eps = result["epsilon"]
    ax1.axvline(1 - eps, color="k", linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.axvline(1 + eps, color="k", linestyle="--", linewidth=1.0, alpha=0.5,
                label=f"1 ± ε = 1 ± {eps}")
    ax1.axvline(1.0, color="k", linewidth=1.0, alpha=0.3)
    ax1.set_xlabel("Distance Ratio (projected / original)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Distance Preservation (k = {max(result['d_targets'])})",
                  fontweight="bold")
    ax1.legend(fontsize=8)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    # Right: max distortion vs target dimension
    ks = result["d_targets"]
    distortions = [result["max_distortions"][k] for k in ks]
    ax2.semilogx(ks, distortions, "o-", color="#2ca02c", linewidth=1.5, markersize=5)
    ax2.axhline(eps, color="k", linestyle="--", linewidth=1.0, alpha=0.5,
                label=f"ε = {eps}")
    ax2.axvline(result["theoretical_k"], color="gray", linestyle=":",
                linewidth=1.0, alpha=0.5,
                label=f"k* = {result['theoretical_k']}")
    ax2.set_xlabel("Target Dimension k")
    ax2.set_ylabel("Max Distortion |ratio - 1|")
    ax2.set_title("Distortion vs. Dimension", fontweight="bold")
    ax2.legend(fontsize=8)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("figures/jl.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Theoretical k for ε={eps}: {result['theoretical_k']}")
