"""
GMM training on Euler angle distribution (yaw, pitch, roll).
Fits sklearn GaussianMixture for multiple K values, selects best by BIC,
saves model and visualizations.
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths
DATA_PATH = "outputs/data/waypoint_orientation_dataset.pt"
MODEL_PATH = "outputs/weights/gmm/euler_gmm.pkl"
PLOT_DIR = "outputs/plots/gmm"
K_CANDIDATES = [2, 3, 5, 8, 10, 15, 20, 30]
LABELS = ["yaw", "pitch", "roll"]


def load_data():
    data = torch.load(DATA_PATH, weights_only=False)
    euler = data["euler_final"].numpy()  # [N, 3]
    print(f"Loaded euler_final: {euler.shape}")
    return euler


def train_gmm_candidates(euler):
    results = {}
    for k in K_CANDIDATES:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(euler)
        bic = gmm.bic(euler)
        results[k] = {"model": gmm, "bic": bic}
        print(f"  K={k:3d}  BIC={bic:12.1f}")
    return results


def select_best(results):
    best_k = min(results, key=lambda k: results[k]["bic"])
    best_model = results[best_k]["model"]
    print(f"\nBest K = {best_k}  (BIC = {results[best_k]['bic']:.1f})")
    print(f"Component weights: {np.round(best_model.weights_, 4)}")
    return best_k, best_model


def plot_bic(results):
    ks = sorted(results.keys())
    bics = [results[k]["bic"] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(k) for k in ks], bics, color="steelblue")
    best_idx = np.argmin(bics)
    bars[best_idx].set_color("tomato")
    ax.set_xlabel("Number of Components (K)")
    ax.set_ylabel("BIC")
    ax.set_title("GMM BIC Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "bic_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved BIC comparison plot")


def gmm_marginal_2d(gmm, dim_i, dim_j):
    """Extract 2D marginal parameters from full 3D GMM."""
    dims = [dim_i, dim_j]
    means_2d = gmm.means_[:, dims]
    covs_2d = gmm.covariances_[:, dims][:, :, dims]
    return means_2d, covs_2d, gmm.weights_


def plot_gmm_fit(euler, gmm):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Pairwise scatter + contour
    pairs = [(0, 1), (0, 2), (1, 2)]
    for col, (i, j) in enumerate(pairs):
        ax = axes[0, col]
        ax.scatter(euler[:, i], euler[:, j], s=1, alpha=0.3, c="gray")

        means_2d, covs_2d, weights = gmm_marginal_2d(gmm, i, j)
        x_range = np.linspace(euler[:, i].min() - 0.3, euler[:, i].max() + 0.3, 100)
        y_range = np.linspace(euler[:, j].min() - 0.3, euler[:, j].max() + 0.3, 100)
        X, Y = np.meshgrid(x_range, y_range)
        pos = np.column_stack([X.ravel(), Y.ravel()])

        Z = np.zeros(pos.shape[0])
        for k in range(len(weights)):
            from scipy.stats import multivariate_normal
            rv = multivariate_normal(means_2d[k], covs_2d[k])
            Z += weights[k] * rv.pdf(pos)
        Z = Z.reshape(X.shape)

        ax.contour(X, Y, Z, levels=8, colors="red", linewidths=0.8, alpha=0.7)
        ax.set_xlabel(f"{LABELS[i]} (rad)")
        ax.set_ylabel(f"{LABELS[j]} (rad)")
        ax.set_title(f"{LABELS[i]} vs {LABELS[j]}")

    # Row 2: 1D marginal histograms + GMM fit
    for col in range(3):
        ax = axes[1, col]
        ax.hist(euler[:, col], bins=80, density=True, alpha=0.5, color="steelblue", label="data")

        x = np.linspace(euler[:, col].min() - 0.3, euler[:, col].max() + 0.3, 500)
        pdf = np.zeros_like(x)
        for k in range(gmm.n_components):
            mu = gmm.means_[k, col]
            sigma = np.sqrt(gmm.covariances_[k, col, col])
            w = gmm.weights_[k]
            component_pdf = w * norm.pdf(x, mu, sigma)
            pdf += component_pdf
            ax.plot(x, component_pdf, "--", alpha=0.4, linewidth=0.8)
        ax.plot(x, pdf, "r-", linewidth=1.5, label="GMM fit")
        ax.set_xlabel(f"{LABELS[col]} (rad)")
        ax.set_ylabel("Density")
        ax.set_title(f"{LABELS[col]} marginal")
        ax.legend(fontsize=8)

    fig.suptitle(f"GMM Fit (K={gmm.n_components})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "gmm_fit.png"), dpi=150)
    plt.close(fig)
    print(f"Saved GMM fit plot")


def main():
    euler = load_data()

    print("\nTraining GMM for each K candidate:")
    results = train_gmm_candidates(euler)

    best_k, best_model = select_best(results)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nSaved best model to {MODEL_PATH}")

    # Plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_bic(results)
    plot_gmm_fit(euler, best_model)

    # Quick sampling test
    samples = best_model.sample(10)[0]
    print(f"\nSample check - 3 random samples (yaw, pitch, roll in rad):")
    for s in samples[:3]:
        print(f"  [{s[0]:+.4f}, {s[1]:+.4f}, {s[2]:+.4f}]")


if __name__ == "__main__":
    main()
