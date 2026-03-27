"""Random Fourier Features: Bochner's theorem made practical.

Rahimi & Recht (2007) showed that for any shift-invariant PD kernel k(x-y),
the map z(x) = √(2/D) [cos(ω₁·x + b₁), ..., cos(ω_D·x + b_D)] with
ωᵢ ~ p(ω) (the spectral measure) satisfies z(x)·z(y) ≈ k(x,y). This turns
kernel methods into linear methods with random features.

Reference: Rahimi & Recht, "Random Features for Large-Scale Kernel Machines",
NeurIPS 2007. (NeurIPS 2017 Test of Time Award.)
"""

import numpy as np


def _sample_rff(
    d: int, D: int, gamma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample Random Fourier Feature parameters.

    For the RBF kernel k(x,y) = exp(-γ‖x-y‖²), the spectral measure is
    N(0, 2γ·I). Sample D frequencies and D uniform biases.

    Args:
        d: Input dimension.
        D: Number of random features.
        gamma: RBF kernel bandwidth parameter.
        rng: Random number generator.

    Returns:
        omega: Sampled frequencies, shape (d, D).
        b: Sampled biases, shape (D,).
    """
    omega = rng.standard_normal((d, D)) * np.sqrt(2 * gamma)
    b = rng.uniform(0, 2 * np.pi, size=D)
    return omega, b


def _rff_features(
    X: np.ndarray, omega: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Compute Random Fourier Feature map.

    z(x) = √(2/D) cos(x·ω + b)

    Args:
        X: Input data, shape (n, d).
        omega: Frequencies, shape (d, D).
        b: Biases, shape (D,).

    Returns:
        Feature matrix Z, shape (n, D).
    """
    D = omega.shape[1]
    return np.sqrt(2.0 / D) * np.cos(X @ omega + b[None, :])


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """Exact RBF kernel matrix: K[i,j] = exp(-γ ‖x_i - x_j‖²)."""
    sq1 = np.sum(X1**2, axis=1)
    sq2 = np.sum(X2**2, axis=1)
    dists_sq = sq1[:, None] + sq2[None, :] - 2 * X1 @ X2.T
    return np.exp(-gamma * dists_sq)


def demonstrate(
    D_values: list[int] | None = None,
    n_train: int = 100,
    n_test: int = 500,
    gamma: float = 1.0,
    ridge: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Demonstrate RFF kernel approximation and regression.

    Args:
        D_values: Numbers of random features. Default: [5, 10, 25, 50, 100, 500].
        n_train: Number of training points.
        n_test: Number of test points.
        gamma: RBF kernel bandwidth.
        ridge: Ridge regularization parameter.
        seed: Random seed.

    Returns:
        Dict with keys: 'x_train', 'y_train', 'x_test', 'y_test', 'f_true',
        'D_values', 'predictions', 'test_errors', 'kernel_errors', 'exact_prediction'.
    """
    if D_values is None:
        D_values = [5, 10, 25, 50, 100, 500]

    rng = np.random.default_rng(seed)

    # 1D regression: f(x) = sin(3x) + 0.5·cos(7x) + noise
    x_train = rng.uniform(0, 2 * np.pi, (n_train, 1))
    x_test = np.linspace(0, 2 * np.pi, n_test).reshape(-1, 1)
    f = lambda t: np.sin(3 * t) + 0.5 * np.cos(7 * t)
    y_train = f(x_train).ravel() + 0.1 * rng.standard_normal(n_train)
    f_true = f(x_test).ravel()

    # Exact kernel regression for reference
    K_train = _rbf_kernel(x_train, x_train, gamma)
    K_test = _rbf_kernel(x_test, x_train, gamma)
    alpha_exact = np.linalg.solve(
        K_train + ridge * np.eye(n_train), y_train
    )
    exact_pred = K_test @ alpha_exact

    predictions = {}
    test_errors = {}
    kernel_errors = {}

    for D in D_values:
        omega, b = _sample_rff(1, D, gamma, rng)
        Z_train = _rff_features(x_train, omega, b)
        Z_test = _rff_features(x_test, omega, b)

        # Ridge regression in feature space
        w = np.linalg.solve(
            Z_train.T @ Z_train + ridge * np.eye(D), Z_train.T @ y_train
        )
        pred = Z_test @ w

        predictions[D] = pred
        test_errors[D] = np.sqrt(np.mean((f_true - pred) ** 2))

        # Kernel approximation error: ‖K - ZZ^T‖_F / ‖K‖_F
        K_approx = Z_train @ Z_train.T
        kernel_errors[D] = (
            np.linalg.norm(K_train - K_approx, "fro")
            / np.linalg.norm(K_train, "fro")
        )

    return {
        "x_train": x_train.ravel(),
        "y_train": y_train,
        "x_test": x_test.ravel(),
        "f_true": f_true,
        "D_values": D_values,
        "predictions": predictions,
        "test_errors": test_errors,
        "kernel_errors": kernel_errors,
        "exact_prediction": exact_pred,
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

    # Left: regression curves
    ax1.scatter(result["x_train"], result["y_train"], s=8, alpha=0.3,
                color="gray", label="Train data")
    ax1.plot(result["x_test"], result["f_true"], "k-", linewidth=1.5,
             label="True f(x)")
    for D in [10, 50, 500]:
        ax1.plot(result["x_test"], result["predictions"][D], "--",
                 linewidth=1.2, label=f"D = {D}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("RFF Regression", fontweight="bold")
    ax1.legend(fontsize=8)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    # Right: kernel approximation error
    Ds = result["D_values"]
    kerr = [result["kernel_errors"][D] for D in Ds]
    ax2.loglog(Ds, kerr, "o-", color="#d62728", linewidth=1.5, markersize=5)
    ax2.set_xlabel("Number of Features D")
    ax2.set_ylabel("Relative Kernel Error")
    ax2.set_title("Kernel Approximation", fontweight="bold")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("figures/rff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Test RMSE at D=500: {result['test_errors'][500]:.4f}")
