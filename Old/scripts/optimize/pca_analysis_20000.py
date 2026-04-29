"""
PCA analysis on 20000-trajectory collision-free dataset.

Produces same style of plots as the 5000 version, but suffix "_20000"
to avoid overwriting existing outputs.

Compares: Original (5000) vs Collision-Free (20000).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
SAVE = os.path.join(ROOT, "outputs/results/collision_free_dataset")
SUFFIX = "_20000"


def main():
    print("=== PCA Analysis: 5000 Original vs 20000 Collision-Free ===\n", flush=True)

    # Load
    d_o = np.load(os.path.join(ROOT, "outputs/results/trajectory_dataset/trajectories_5000x100x6.npy"))
    d_f = np.load(os.path.join(SAVE, "trajectories_collision_free_20000x100x6.npy"))
    No, T, D = d_o.shape
    Nf = d_f.shape[0]
    print(f"Original:       {d_o.shape}")
    print(f"Collision-Free: {d_f.shape}")

    # Fit PCA independently on each set, using all (traj, timestep) as samples
    flat_o = d_o.reshape(No * T, D)
    flat_f = d_f.reshape(Nf * T, D)

    pca_o = PCA(n_components=D).fit(flat_o)
    pca_f = PCA(n_components=D).fit(flat_f)

    emb_o = pca_o.transform(flat_o).reshape(No, T, D)
    emb_f = pca_f.transform(flat_f).reshape(Nf, T, D)

    print(f"\nOriginal PCA variance ratio:")
    print(f"  {pca_o.explained_variance_ratio_[:4].round(4)}")
    print(f"  Top 2 total: {pca_o.explained_variance_ratio_[:2].sum()*100:.1f}%")
    print(f"Col-Free PCA variance ratio:")
    print(f"  {pca_f.explained_variance_ratio_[:4].round(4)}")
    print(f"  Top 2 total: {pca_f.explained_variance_ratio_[:2].sum()*100:.1f}%")

    # Save 2D embedding (for downstream)
    emb_f_2d = emb_f[:, :, :2]
    emb_path = os.path.join(SAVE, f"pca_embedded_20000x100x2.npy")
    np.save(emb_path, emb_f_2d)
    print(f"\nSaved: {emb_path}  shape={emb_f_2d.shape}")

    rng = np.random.default_rng(42)
    t_axis = np.linspace(0, 10, T)

    # ---- Plot 1: 2D snapshots at multiple timesteps ----
    snapshot_ts = [0, 10, 25, 50, 75, 99]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=False, sharey=False)
    for ax, ts in zip(axes.flat, snapshot_ts):
        ax.scatter(emb_o[:, ts, 0], emb_o[:, ts, 1], s=3, alpha=0.3,
                   color='tab:red', label=f'Orig N={No}')
        ax.scatter(emb_f[:, ts, 0], emb_f[:, ts, 1], s=2, alpha=0.25,
                   color='tab:blue', label=f'Col-Free N={Nf}')
        ax.set_title(f"t={ts*0.1:.1f}s", fontsize=11)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        if ts == snapshot_ts[0]:
            ax.legend(fontsize=8, loc='best')
    plt.suptitle(f"PCA 2D Snapshots: Original (5000) vs Collision-Free ({Nf})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE, f"pca_comparison_2d_snapshots{SUFFIX}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

    # ---- Plot 2: 3D comparison (PC1, PC2, time), side-by-side ----
    fig = plt.figure(figsize=(18, 7))
    for panel, (emb, N, title, color) in enumerate([
        (emb_o, No, f"Original (N={No})", 'tab:red'),
        (emb_f, Nf, f"Collision-Free (N={Nf})", 'tab:blue'),
    ]):
        ax = fig.add_subplot(1, 2, panel + 1, projection='3d')
        n_show = min(400, N)
        idx = rng.choice(N, n_show, replace=False)
        for i in idx:
            ax.plot(emb[i, :, 0], emb[i, :, 1], t_axis,
                    alpha=0.15, linewidth=0.6, color=color)
        # Highlight a few
        hl_colors = ['#e41a1c', '#ff7f00', '#4daf4a']
        for k in range(3):
            ax.plot(emb[idx[k], :, 0], emb[idx[k], :, 1], t_axis,
                    alpha=0.9, linewidth=2.2, color=hl_colors[k])
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("Time (s)")
        ax.set_title(title, fontsize=12)
        ax.view_init(elev=25, azim=-60)
    plt.suptitle(f"PCA 3D: Original vs Collision-Free ({Nf})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(SAVE, f"pca_comparison_3d{SUFFIX}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

    # ---- Plot 3: 3D timeflow (col-free only, dense) ----
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    n_show = 800
    idx = rng.choice(Nf, n_show, replace=False)
    for i in idx:
        ax.plot(emb_f[i, :, 0], emb_f[i, :, 1], t_axis,
                alpha=0.25, linewidth=0.8, color='tab:blue')
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("Time (s)")
    ax.set_title(f"Collision-Free (N={Nf}) — PC1-PC2-Time flow", fontsize=12)
    ax.view_init(elev=22, azim=-55)
    plt.tight_layout()
    path = os.path.join(SAVE, f"pca_3d_timeflow{SUFFIX}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

    # ---- Plot 4: Variance comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(1, D + 1); w = 0.35
    ax.bar(x - w/2, pca_o.explained_variance_ratio_ * 100, w,
           label=f'Orig (N={No})', color='tab:red')
    ax.bar(x + w/2, pca_f.explained_variance_ratio_ * 100, w,
           label=f'Col-Free (N={Nf})', color='tab:blue')
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance (%)")
    ax.set_title("Variance per Component"); ax.set_xticks(x)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    cum_o = np.cumsum(pca_o.explained_variance_ratio_) * 100
    cum_f = np.cumsum(pca_f.explained_variance_ratio_) * 100
    ax.plot(x, cum_o, 'o-', label=f'Orig (N={No})', color='tab:red')
    ax.plot(x, cum_f, 's-', label=f'Col-Free (N={Nf})', color='tab:blue')
    ax.axhline(90, color='gray', linestyle='--', linewidth=0.8, label='90%')
    ax.set_xlabel("# Components"); ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("Cumulative Variance"); ax.set_xticks(x)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f"PCA Variance: Original vs Collision-Free ({Nf})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE, f"pca_variance_comparison{SUFFIX}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

    # ---- Plot 5: Loadings comparison (heatmap) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (pca_obj, title) in zip(axes, [
        (pca_o, f"Original (N={No})"),
        (pca_f, f"Collision-Free (N={Nf})"),
    ]):
        im = ax.imshow(pca_obj.components_, aspect='auto',
                       cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_yticks(range(D))
        ax.set_yticklabels([f"PC{i+1}" for i in range(D)])
        ax.set_xticks(range(D))
        ax.set_xticklabels([f"J{i+1}" for i in range(D)])
        ax.set_title(title, fontsize=11)
        for i in range(D):
            for j in range(D):
                v = pca_obj.components_[i, j]
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > 0.5 else 'black')
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.suptitle(f"PCA Loadings: Original vs Collision-Free ({Nf})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE, f"pca_loadings_comparison{SUFFIX}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

    # ---- Joint range comparison ----
    print(f"\n--- Joint range (rad) ---")
    print(f"{'Joint':<6} {'Original':>24} {'Col-Free (20000)':>28}")
    for j in range(D):
        o = d_o[:, :, j]; f = d_f[:, :, j]
        print(f"  J{j+1}   [{o.min():+.3f}, {o.max():+.3f}]"
              f"        [{f.min():+.3f}, {f.max():+.3f}]")

    print(f"\n=== Done. Output dir: {SAVE} ===")


if __name__ == "__main__":
    main()
