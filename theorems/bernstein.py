"""Weierstrass approximation via Bernstein polynomials.

Bernstein (1912) gave a constructive proof of Weierstrass's theorem: for any
continuous f: [0,1] → R, the polynomial B_n(f; x) = Σ f(k/n) C(n,k) x^k (1-x)^{n-k}
converges uniformly to f. The construction is probabilistic: B_n(f; x) = E[f(S_n/n)]
where S_n ~ Binomial(n, x).

Reference: Bernstein, "Démonstration du théorème de Weierstrass fondée sur le calcul
des probabilités", Comm. Kharkov Math. Soc., 1912.
"""

from collections.abc import Callable

import numpy as np
from scipy.special import comb


def _bernstein_basis(n: int, k: int, x: np.ndarray) -> np.ndarray:
    """Evaluate the k-th Bernstein basis polynomial of degree n.

    B_{k,n}(x) = C(n,k) x^k (1-x)^{n-k}

    This is exactly the Binomial(n, x) probability mass at k — the bridge
    between approximation theory and probability.
    """
    return comb(n, k, exact=True) * x**k * (1 - x) ** (n - k)


def bernstein_approximate(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, n: int
) -> np.ndarray:
    """Construct the degree-n Bernstein polynomial approximation of f.

    Args:
        f: Target function, maps [0,1] → R.
        x: Evaluation points in [0,1], shape (N,).
        n: Polynomial degree.

    Returns:
        B_n(f; x), shape (N,).
    """
    # Evaluate f at the equispaced nodes k/n
    nodes = np.arange(n + 1) / n
    f_values = f(nodes)

    # B_n(f; x) = Σ_{k=0}^{n} f(k/n) B_{k,n}(x)
    result = np.zeros_like(x, dtype=float)
    for k in range(n + 1):
        result += f_values[k] * _bernstein_basis(n, k, x)
    return result


def demonstrate(
    f: Callable[[np.ndarray], np.ndarray] | None = None,
    x: np.ndarray | None = None,
    degrees: list[int] | None = None,
) -> dict:
    """Construct Bernstein polynomial approximations of f at given degrees.

    Args:
        f: Target function, maps [0,1] → R. Default: sin(5πx)·exp(-2x) + 0.5.
        x: Evaluation points, shape (N,). Default: 1000 points in [0,1].
        degrees: Polynomial degrees to demonstrate. Default: [5, 10, 25, 50, 100].

    Returns:
        Dict with keys: 'x', 'f_true', 'degrees', 'approximations', 'max_errors'.
    """
    if f is None:
        f = lambda t: np.sin(5 * np.pi * t) * np.exp(-2 * t) + 0.5
    if x is None:
        x = np.linspace(0, 1, 1000)
    if degrees is None:
        degrees = [5, 10, 25, 50, 100]

    f_true = f(x)

    approximations = {}
    max_errors = {}

    for n in degrees:
        approx = bernstein_approximate(f, x, n)
        approximations[n] = approx
        max_errors[n] = np.max(np.abs(f_true - approx))

    return {
        "x": x,
        "f_true": f_true,
        "degrees": degrees,
        "approximations": approximations,
        "max_errors": max_errors,
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

    # Left: approximations
    ax1.plot(result["x"], result["f_true"], "k-", linewidth=1.5, label="f(x)")
    cmap = plt.cm.Blues
    for i, n in enumerate(result["degrees"]):
        color = cmap(0.3 + 0.7 * i / (len(result["degrees"]) - 1))
        ax1.plot(result["x"], result["approximations"][n], "--",
                 color=color, linewidth=1.2, label=f"n = {n}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("Bernstein Polynomial Approximation", fontweight="bold")
    ax1.legend(fontsize=8)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    # Right: convergence
    ns = result["degrees"]
    errs = [result["max_errors"][n] for n in ns]
    ax2.loglog(ns, errs, "o-", color="#1f77b4", linewidth=1.5, markersize=5)
    ax2.loglog(ns, errs[0] * np.sqrt(ns[0] / np.array(ns, dtype=float)),
               "k--", linewidth=1.0, alpha=0.5, label=r"$O(1/\sqrt{n})$")
    ax2.set_xlabel("Degree n")
    ax2.set_ylabel("Max Error")
    ax2.set_title("Convergence Rate", fontweight="bold")
    ax2.legend(fontsize=8)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("figures/bernstein.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Max error at n=100: {result['max_errors'][100]:.4f}")
