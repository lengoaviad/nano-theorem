"""Chen & Chen universal approximation for operators via DeepONet.

Chen & Chen (1995) proved that neural networks can approximate any continuous
nonlinear operator G: V → C(K₂), where V is compact in C(K₁). The architecture:
G(u)(y) ≈ Σ_k branch_k(u(x₁),...,u(x_m)) · trunk_k(y). This is the theoretical
foundation of DeepONet (Lu et al., 2021).

References:
- Chen & Chen, "Universal approximation to nonlinear operators by neural
  networks with arbitrary activation functions", IEEE Trans. Neural Networks, 1995.
- Lu et al., "Learning nonlinear operators via DeepONet", Nature Machine
  Intelligence, 2021.
"""

import numpy as np
import torch
import torch.nn as nn


class _BranchNet(nn.Module):
    """Branch network: maps sensor values u(x₁),...,u(x_m) to R^p."""

    def __init__(self, m_sensors: int, hidden: int, p: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m_sensors, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, p),
        )

    def forward(self, u_sensors: torch.Tensor) -> torch.Tensor:
        return self.net(u_sensors)


class _TrunkNet(nn.Module):
    """Trunk network: maps evaluation point y to R^p."""

    def __init__(self, hidden: int, p: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, p),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class _DeepONet(nn.Module):
    """DeepONet: dot product of branch and trunk outputs.

    G(u)(y) ≈ Σ_k branch_k(u_sensors) · trunk_k(y) + bias
    """

    def __init__(self, m_sensors: int, hidden: int = 40, p: int = 40):
        super().__init__()
        self.branch = _BranchNet(m_sensors, hidden, p)
        self.trunk = _TrunkNet(hidden, p)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, u_sensors: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        b = self.branch(u_sensors)  # (batch, p)
        t = self.trunk(y)  # (batch, p)
        return torch.sum(b * t, dim=-1, keepdim=True) + self.bias


def _generate_antiderivative_data(
    n_samples: int, m_sensors: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate random functions and their antiderivatives.

    Input functions: random piecewise-linear on [0, 1] with m_sensors nodes.
    Operator: G(u)(y) = ∫₀ʸ u(s) ds (antiderivative).

    Args:
        n_samples: Number of random input functions.
        m_sensors: Number of sensor points (also function nodes).
        rng: Random number generator.

    Returns:
        u_sensors: Input function values at sensors, (n_samples, m_sensors).
        y_eval: Evaluation points, (n_samples, 1).
        Gu_true: True antiderivative values, (n_samples, 1).
        sensor_locations: Sensor positions in [0, 1], (m_sensors,).
    """
    sensor_locations = np.linspace(0, 1, m_sensors)

    u_sensors = np.zeros((n_samples, m_sensors))
    y_eval = np.zeros((n_samples, 1))
    Gu_true = np.zeros((n_samples, 1))

    for i in range(n_samples):
        # Random piecewise-linear function: sample values at sensors
        u_vals = rng.standard_normal(m_sensors)
        u_sensors[i] = u_vals

        # Random evaluation point
        y = rng.uniform(0, 1)
        y_eval[i, 0] = y

        # Antiderivative via trapezoidal integration up to y
        # Interpolate u at a fine grid, then integrate
        fine_grid = np.linspace(0, y, 200)
        u_fine = np.interp(fine_grid, sensor_locations, u_vals)
        Gu_true[i, 0] = np.trapezoid(u_fine, fine_grid)

    return u_sensors, y_eval, Gu_true, sensor_locations


def demonstrate(
    n_train: int = 5000,
    n_test: int = 500,
    m_sensors: int = 50,
    epochs: int = 2000,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train a DeepONet to learn the antiderivative operator.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        m_sensors: Number of sensor points.
        epochs: Training epochs.
        lr: Learning rate.
        seed: Random seed.

    Returns:
        Dict with keys: 'sensor_locations', 'test_u', 'test_y', 'test_true',
        'test_pred', 'train_losses', 'test_error'.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Generate data
    u_train, y_train, Gu_train, sensors = _generate_antiderivative_data(
        n_train, m_sensors, rng
    )
    u_test, y_test, Gu_test, _ = _generate_antiderivative_data(
        n_test, m_sensors, rng
    )

    # Convert to tensors
    u_train_t = torch.tensor(u_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    Gu_train_t = torch.tensor(Gu_train, dtype=torch.float32)
    u_test_t = torch.tensor(u_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Build and train model
    model = _DeepONet(m_sensors, hidden=40, p=40)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []

    for epoch in range(epochs):
        model.train()
        pred = model(u_train_t, y_train_t)
        loss = loss_fn(pred, Gu_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            train_losses.append((epoch, loss.item()))

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_pred = model(u_test_t, y_test_t).numpy()

    test_error = np.sqrt(np.mean((Gu_test - test_pred) ** 2))

    # Select 3 example functions for visualization
    # Pick indices with diverse y values for visual interest
    sort_idx = np.argsort(y_test[:, 0])
    example_indices = [sort_idx[n_test // 4], sort_idx[n_test // 2],
                       sort_idx[3 * n_test // 4]]

    # For each example, compute full antiderivative curve
    example_data = []
    model.eval()
    for idx in example_indices:
        u_vals = u_test[idx]
        y_fine = np.linspace(0, 1, 200)

        # True antiderivative at each point
        Gu_fine = np.zeros(200)
        for j, yj in enumerate(y_fine):
            fine_grid = np.linspace(0, yj, 200)
            u_fine = np.interp(fine_grid, sensors, u_vals)
            Gu_fine[j] = np.trapezoid(u_fine, fine_grid)

        # Predicted antiderivative
        with torch.no_grad():
            u_rep = torch.tensor(
                np.tile(u_vals, (200, 1)), dtype=torch.float32
            )
            y_rep = torch.tensor(
                y_fine.reshape(-1, 1), dtype=torch.float32
            )
            pred_fine = model(u_rep, y_rep).numpy().ravel()

        example_data.append({
            "u_sensors": u_vals,
            "y_fine": y_fine,
            "Gu_true": Gu_fine,
            "Gu_pred": pred_fine,
        })

    return {
        "sensor_locations": sensors,
        "example_data": example_data,
        "train_losses": train_losses,
        "test_error": test_error,
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

    result = demonstrate(epochs=1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: example predictions
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (ex, color) in enumerate(zip(result["example_data"], colors)):
        ax1.plot(ex["y_fine"], ex["Gu_true"], "-", color=color, linewidth=1.5,
                 label=f"True #{i+1}")
        ax1.plot(ex["y_fine"], ex["Gu_pred"], "--", color=color, linewidth=1.2)
    ax1.set_xlabel("y")
    ax1.set_ylabel("G(u)(y)")
    ax1.set_title("DeepONet: Antiderivative Operator", fontweight="bold")
    ax1.legend(fontsize=7, ncol=2)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    # Right: training loss
    epochs_arr, losses = zip(*result["train_losses"])
    ax2.semilogy(epochs_arr, losses, "-", color="#8c564b", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Training Loss", fontweight="bold")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("figures/deeponet.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Test RMSE: {result['test_error']:.4f}")
