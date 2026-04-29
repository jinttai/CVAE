"""
PCA analysis on trajectory dataset [5000, 100, 6].

Approach:
  1. Stack all (trajectory, timestep) pairs → [500000, 6]
  2. Fit PCA (6D → 2D) on shared basis
  3. Reshape back to [5000, 100, 2]
  4. 3D plot: (PC1, PC2, timestep) — each trajectory is a curve
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))

def main():
    # ---- Load ----
    data_path = os.path.join(ROOT_DIR, "outputs/results/trajectory_dataset/trajectories_5000x100x6.npy")
    save_dir = os.path.join(ROOT_DIR, "outputs/results/trajectory_dataset")
    data = np.load(data_path)  # [5000, 100, 6]
    N, T, D = data.shape
    print(f"Loaded: {data.shape}  ({N} trajectories, {T} timesteps, {D} joints)")

    # ---- PCA: fit on all (traj, time) pairs ----
    flat = data.reshape(N * T, D)  # [500000, 6]
    pca = PCA(n_components=2)
    flat_2d = pca.fit_transform(flat)  # [500000, 2]
    embedded = flat_2d.reshape(N, T, 2)  # [5000, 100, 2]

    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"  Total: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"\nPCA components (loadings):")
    for i, comp in enumerate(pca.components_):
        joints = "  ".join([f"J{j+1}:{v:+.3f}" for j, v in enumerate(comp)])
        print(f"  PC{i+1}: {joints}")

    t_axis = np.linspace(0, 10, T)

    # ---- Plot 1: 3D (PC1, PC2, time) — all trajectories ----
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Sample trajectories for clarity
    n_show = 200
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=n_show, replace=False)

    for i in idx:
        ax.plot(embedded[i, :, 0], embedded[i, :, 1], t_axis,
                alpha=0.15, linewidth=0.5, color='tab:blue')

    # Highlight a few
    for k, i in enumerate(idx[:5]):
        ax.plot(embedded[i, :, 0], embedded[i, :, 1], t_axis,
                alpha=0.9, linewidth=1.5, label=f"Traj {i}")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Time (s)")
    ax.set_title(f"Trajectory PCA Embedding (PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
                 f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path1 = os.path.join(save_dir, "pca_3d_trajectories.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"\nSaved: {path1}")

    # ---- Plot 2: 3D with time colormap ----
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in idx[:100]:
        points = embedded[i]  # [100, 2]
        colors = plt.cm.viridis(np.linspace(0, 1, T))
        for t in range(T - 1):
            ax.plot(points[t:t+2, 0], points[t:t+2, 1], t_axis[t:t+2],
                    color=colors[t], alpha=0.3, linewidth=0.6)

    # Add start/end markers for a few
    for i in idx[:10]:
        ax.scatter(*embedded[i, 0, :], t_axis[0], c='green', s=20, zorder=5)
        ax.scatter(*embedded[i, -1, :], t_axis[-1], c='red', s=20, zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Time (s)")
    ax.set_title("Trajectory Flow in PCA Space (color = time)")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 10))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Time (s)", shrink=0.6)

    plt.tight_layout()
    path2 = os.path.join(save_dir, "pca_3d_timecolor.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved: {path2}")

    # ---- Plot 3: 2D snapshots at key timesteps ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    timesteps = [0, 10, 25, 50, 75, 99]
    titles = ["t=0.0s (start)", "t=1.0s", "t=2.5s", "t=5.0s (mid)", "t=7.5s", "t=9.9s (end)"]

    for ax_idx, (ts, title) in enumerate(zip(timesteps, titles)):
        ax = axes[ax_idx // 3, ax_idx % 3]
        x = embedded[:, ts, 0]
        y = embedded[:, ts, 1]
        ax.scatter(x, y, s=2, alpha=0.3, c='tab:blue')
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(embedded[:, :, 0].min() - 0.5, embedded[:, :, 0].max() + 0.5)
        ax.set_ylim(embedded[:, :, 1].min() - 0.5, embedded[:, :, 1].max() + 0.5)

    plt.suptitle("PCA 2D Snapshots at Key Timesteps", fontsize=14)
    plt.tight_layout()
    path3 = os.path.join(save_dir, "pca_2d_snapshots.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"Saved: {path3}")

    # ---- Plot 4: Variance explained (full spectrum) ----
    pca_full = PCA(n_components=D)
    pca_full.fit(flat)
    var_ratio = pca_full.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.bar(range(1, D + 1), var_ratio * 100, color='tab:blue')
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Variance per Component")
    ax.set_xticks(range(1, D + 1))
    ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(var_ratio):
        ax.text(i + 1, v * 100 + 0.5, f"{v*100:.1f}%", ha='center', fontsize=9)

    ax = axes[1]
    cumvar = np.cumsum(var_ratio) * 100
    ax.plot(range(1, D + 1), cumvar, 'o-', color='tab:orange')
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("Cumulative Explained Variance")
    ax.set_xticks(range(1, D + 1))
    ax.axhline(90, color='r', linestyle='--', linewidth=0.8, label='90%')
    ax.axhline(95, color='gray', linestyle='--', linewidth=0.8, label='95%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path4 = os.path.join(save_dir, "pca_variance.png")
    plt.savefig(path4, dpi=150)
    plt.close()
    print(f"Saved: {path4}")

    # ---- Plot 5: PCA loadings heatmap ----
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pca_full.components_, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(D))
    ax.set_yticklabels([f"PC{i+1}" for i in range(D)])
    ax.set_xticks(range(D))
    ax.set_xticklabels([f"J{i+1}" for i in range(D)])
    ax.set_title("PCA Loadings (Component × Joint)")
    plt.colorbar(im, ax=ax)
    for i in range(D):
        for j in range(D):
            ax.text(j, i, f"{pca_full.components_[i, j]:.2f}",
                    ha='center', va='center', fontsize=9,
                    color='white' if abs(pca_full.components_[i, j]) > 0.5 else 'black')
    plt.tight_layout()
    path5 = os.path.join(save_dir, "pca_loadings.png")
    plt.savefig(path5, dpi=150)
    plt.close()
    print(f"Saved: {path5}")

    # ---- Save embedded data for future use ----
    np.save(os.path.join(save_dir, "pca_embedded_5000x100x2.npy"), embedded.astype(np.float32))
    print(f"\nSaved embedded data: {embedded.shape}")


if __name__ == "__main__":
    main()
