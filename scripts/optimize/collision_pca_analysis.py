"""
Self-collision 필터링 후 PCA 비교 분석.

1. 기존 5000 trajectory에 MuJoCo collision check
2. Collision-free만 필터링
3. 원본 vs collision-free PCA 비교 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA
import os
import sys
import time

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.utils.collision import CollisionChecker


def check_all_trajectories(data, checker):
    """
    [N, T, 6] trajectory 전체에 대해 collision check.
    Returns:
        collision_free_mask: [N] bool — True if NO collision at any timestep
        collision_count: [N] int — number of colliding timesteps per trajectory
    """
    N, T, _ = data.shape
    collision_count = np.zeros(N, dtype=int)

    t0 = time.time()
    for i in range(N):
        for t in range(T):
            if checker.check(data[i, t]):
                collision_count[i] += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"  {i+1}/{N}  ({elapsed:.1f}s, ETA {eta:.0f}s)  "
                  f"collision-free so far: {(collision_count[:i+1]==0).sum()}", flush=True)

    elapsed = time.time() - t0
    collision_free_mask = collision_count == 0
    print(f"  Done: {N} trajectories in {elapsed:.1f}s", flush=True)
    return collision_free_mask, collision_count


def main():
    print("=== Self-Collision Filtering + PCA Comparison ===\n", flush=True)

    save_dir = os.path.join(ROOT_DIR, "outputs/results/collision_analysis")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Load data ----
    data_path = os.path.join(ROOT_DIR, "outputs/results/trajectory_dataset/trajectories_5000x100x6.npy")
    data = np.load(data_path)
    N, T, D = data.shape
    print(f"Loaded: {data.shape}", flush=True)

    # ---- Step 1: Collision check ----
    print("\n--- Collision Check ---", flush=True)
    checker = CollisionChecker(os.path.join(ROOT_DIR, "assets/spacerobot_collision.xml"))
    collision_free_mask, collision_count = check_all_trajectories(data, checker)

    n_free = collision_free_mask.sum()
    n_colliding = N - n_free
    print(f"\nCollision-free: {n_free}/{N} ({n_free/N*100:.1f}%)")
    print(f"Colliding:      {n_colliding}/{N} ({n_colliding/N*100:.1f}%)")

    # Collision severity distribution
    colliding_counts = collision_count[collision_count > 0]
    if len(colliding_counts) > 0:
        print(f"\nColliding trajectories - timesteps with collision:")
        print(f"  mean: {colliding_counts.mean():.1f} / {T}")
        print(f"  max:  {colliding_counts.max()} / {T}")
        print(f"  1-10 steps:  {((colliding_counts >= 1) & (colliding_counts <= 10)).sum()}")
        print(f"  11-50 steps: {((colliding_counts > 10) & (colliding_counts <= 50)).sum()}")
        print(f"  51+ steps:   {(colliding_counts > 50).sum()}")

    # Save filtered data
    data_free = data[collision_free_mask]
    free_path = os.path.join(save_dir, "trajectories_collision_free.npy")
    np.save(free_path, data_free)
    print(f"\nSaved collision-free: {data_free.shape} → {free_path}")

    # Save stats
    stats_path = os.path.join(save_dir, "collision_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Total trajectories: {N}\n")
        f.write(f"Collision-free: {n_free} ({n_free/N*100:.1f}%)\n")
        f.write(f"Colliding: {n_colliding} ({n_colliding/N*100:.1f}%)\n")
        if len(colliding_counts) > 0:
            f.write(f"Colliding mean timesteps: {colliding_counts.mean():.1f}/{T}\n")
            f.write(f"Colliding max timesteps: {colliding_counts.max()}/{T}\n")

    if n_free < 10:
        print("\nWARNING: Too few collision-free trajectories for meaningful PCA comparison.")
        return

    # ---- Step 2: PCA comparison ----
    print("\n--- PCA Analysis ---", flush=True)

    # Fit PCA independently on each set
    flat_all = data.reshape(N * T, D)
    flat_free = data_free.reshape(n_free * T, D)

    pca_all = PCA(n_components=D)
    pca_free = PCA(n_components=D)

    emb_all_full = pca_all.fit_transform(flat_all).reshape(N, T, D)
    emb_free_full = pca_free.fit_transform(flat_free).reshape(n_free, T, D)

    emb_all = emb_all_full[:, :, :2]    # [N, T, 2]
    emb_free = emb_free_full[:, :, :2]  # [n_free, T, 2]

    print(f"\nOriginal PCA variance:       {pca_all.explained_variance_ratio_[:2]} "
          f"(total {pca_all.explained_variance_ratio_[:2].sum()*100:.1f}%)")
    print(f"Collision-free PCA variance: {pca_free.explained_variance_ratio_[:2]} "
          f"(total {pca_free.explained_variance_ratio_[:2].sum()*100:.1f}%)")

    t_axis = np.linspace(0, 10, T)
    rng = np.random.default_rng(42)

    # ---- Plot 1: 3D comparison side by side ----
    fig = plt.figure(figsize=(20, 8))

    for panel, (emb, title, n_total) in enumerate([
        (emb_all, f"Original ({N} traj)", N),
        (emb_free, f"Collision-Free ({n_free} traj)", n_free),
    ]):
        ax = fig.add_subplot(1, 2, panel + 1, projection='3d')
        n_show = min(200, n_total)
        idx = rng.choice(n_total, n_show, replace=False)
        for i in idx:
            ax.plot(emb[i, :, 0], emb[i, :, 1], t_axis,
                    alpha=0.1, linewidth=0.4, color='steelblue')
        # Highlight 3
        colors = ['#e41a1c', '#ff7f00', '#4daf4a']
        for k in range(min(3, n_show)):
            ax.plot(emb[idx[k], :, 0], emb[idx[k], :, 1], t_axis,
                    alpha=0.9, linewidth=1.8, color=colors[k])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("Time (s)")
        ax.set_title(title, fontsize=12)
        ax.view_init(elev=25, azim=-60)

    plt.suptitle("PCA 3D: Original vs Collision-Free", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_comparison_3d.png"), dpi=150)
    plt.close()
    print("Saved: pca_comparison_3d.png", flush=True)

    # ---- Plot 2: 2D snapshot at t=5s ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ts = 50  # t=5s
    for ax, (emb, title) in zip(axes, [
        (emb_all, f"Original (t=5s, N={N})"),
        (emb_free, f"Collision-Free (t=5s, N={n_free})"),
    ]):
        ax.scatter(emb[:, ts, 0], emb[:, ts, 1], s=3, alpha=0.3, c='tab:blue')
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # Use same axis limits
        lim = max(abs(emb_all[:, ts, :]).max(), abs(emb_free[:, ts, :]).max()) + 0.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    plt.suptitle("PCA 2D Snapshot at t=5s", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_comparison_2d_t5.png"), dpi=150)
    plt.close()
    print("Saved: pca_comparison_2d_t5.png", flush=True)

    # ---- Plot 3: Variance explained comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(1, D + 1)
    w = 0.35
    ax.bar(x - w/2, pca_all.explained_variance_ratio_ * 100, w, label='Original', color='tab:blue')
    ax.bar(x + w/2, pca_free.explained_variance_ratio_ * 100, w, label='Collision-Free', color='tab:green')
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Variance per Component")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    cum_all = np.cumsum(pca_all.explained_variance_ratio_) * 100
    cum_free = np.cumsum(pca_free.explained_variance_ratio_) * 100
    ax.plot(x, cum_all, 'o-', label='Original', color='tab:blue')
    ax.plot(x, cum_free, 's-', label='Collision-Free', color='tab:green')
    ax.axhline(90, color='r', linestyle='--', linewidth=0.8, label='90%')
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("Cumulative Explained Variance")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("PCA Variance: Original vs Collision-Free", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_variance_comparison.png"), dpi=150)
    plt.close()
    print("Saved: pca_variance_comparison.png", flush=True)

    # ---- Plot 4: Loadings heatmap comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (pca_obj, title) in zip(axes, [
        (pca_all, "Original"),
        (pca_free, "Collision-Free"),
    ]):
        im = ax.imshow(pca_obj.components_, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_yticks(range(D))
        ax.set_yticklabels([f"PC{i+1}" for i in range(D)])
        ax.set_xticks(range(D))
        ax.set_xticklabels([f"J{i+1}" for i in range(D)])
        ax.set_title(title)
        for i in range(D):
            for j in range(D):
                v = pca_obj.components_[i, j]
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > 0.5 else 'black')
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.suptitle("PCA Loadings Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_loadings_comparison.png"), dpi=150)
    plt.close()
    print("Saved: pca_loadings_comparison.png", flush=True)

    # ---- Plot 5: Collision heatmap over time ----
    # For colliding trajectories, where in time do collisions happen?
    if n_colliding > 0:
        data_colliding = data[~collision_free_mask]
        n_col = data_colliding.shape[0]
        col_per_timestep = np.zeros(T)
        sample_size = min(n_col, 1000)
        sample_idx = rng.choice(n_col, sample_size, replace=False)
        for i in sample_idx:
            for t in range(T):
                if checker.check(data_colliding[i, t]):
                    col_per_timestep[t] += 1
        col_per_timestep /= sample_size

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.linspace(0, 10, T), col_per_timestep * 100, width=0.1, color='tab:red', alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Collision Rate (%)")
        ax.set_title(f"When Do Collisions Happen? (sampled {sample_size} colliding trajectories)")
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "collision_time_distribution.png"), dpi=150)
        plt.close()
        print("Saved: collision_time_distribution.png", flush=True)

    # ---- Joint range comparison ----
    print(f"\n--- Joint Range Comparison ---")
    print(f"{'Joint':<6} {'Original [min, max]':>24} {'Collision-Free [min, max]':>28}")
    for j in range(D):
        orig = data[:, :, j]
        free = data_free[:, :, j]
        print(f"  J{j+1}   [{orig.min():+.3f}, {orig.max():+.3f}]"
              f"       [{free.min():+.3f}, {free.max():+.3f}]")

    print(f"\n=== Done. Results in {save_dir} ===", flush=True)


if __name__ == "__main__":
    main()
