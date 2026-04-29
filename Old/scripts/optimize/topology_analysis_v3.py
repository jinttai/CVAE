"""
Topology analysis v3 — waypoint-based (decision-variable-based) TDA.

v2: flatten 600D (all timesteps) → trajectory-space topology
v3: extract 3 waypoints × 6 joints = 18D (optimization decision vars only)

Rationale:
  - Quintic interpolation makes intermediate timesteps DEPENDENT on waypoints.
  - The 18D waypoint vector IS the actual free parameter of the optimization.
  - Comparing modes in waypoint space = comparing optimization solution families directly.
  - Much more compact (18D vs 600D), no interpolation redundancy.

Waypoints are at trajectory indices [24, 49, 74] (end of segments 0,1,2; q_start=step 0, q_end=step 99).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time
from ripser import ripser
from persim import plot_diagrams
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
SAVE = os.path.join(ROOT, "outputs/results/topology_analysis_v3")
os.makedirs(SAVE, exist_ok=True)

WP_INDICES = [24, 49, 74]   # end of quintic segments 0, 1, 2


# ------------------------- utils -------------------------

def extract_waypoints(data):
    """data [N, 100, 6] -> waypoints [N, 3, 6] -> flat [N, 18]."""
    wp = data[:, WP_INDICES, :]         # [N, 3, 6]
    return wp, wp.reshape(data.shape[0], -1)


def count_persistent_clusters(h0, gap_ratio=1.5):
    """Use relative drop >= gap_ratio. Default 1.5 (was 1.8 — too strict)."""
    pers = h0[:, 1] - h0[:, 0]
    finite = np.sort(pers[np.isfinite(pers)])[::-1]
    if len(finite) == 0:
        return 1
    k_real = 0
    for i in range(min(20, len(finite) - 1)):
        if finite[i] / max(finite[i + 1], 1e-12) >= gap_ratio:
            k_real = i + 1
    return k_real + 1


def twonn_id(X):
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nbrs.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    valid = (r1 > 1e-10) & (r2 > r1 + 1e-10)
    return np.log(2) / np.log(r2[valid] / r1[valid]).mean()


# ------------------------- analysis -------------------------

def analyze_waypoint_space(data, label, n_sub=1500):
    N = data.shape[0]
    wp_3d, wp_flat = extract_waypoints(data)   # [N,3,6], [N,18]

    print(f"\n=== Waypoint-space {label}: {N} trajectories in 18D ===", flush=True)
    print(f"  Per-joint waypoint range:")
    for j in range(6):
        vals = wp_3d[:, :, j]
        print(f"    joint {j}: [{vals.min():+.2f}, {vals.max():+.2f}]  "
              f"std={vals.std():.2f}")

    # Intrinsic dim in 18D
    id_raw = twonn_id(wp_flat)
    print(f"  Intrinsic dim (TwoNN on raw 18D): {id_raw:.2f}")

    # PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(wp_flat)
    var_cum = pca.explained_variance_ratio_.cumsum()
    print(f"  PCA 10D cumulative variance: {var_cum[-1]:.3f}")
    print(f"  Top 5 variance: {pca.explained_variance_ratio_[:5].round(3)}")

    id_pca = twonn_id(X_pca)
    print(f"  Intrinsic dim (TwoNN on 10D PCA): {id_pca:.2f}")

    # Ripser on subsample
    rng = np.random.default_rng(42)
    sub = rng.choice(N, min(n_sub, N), replace=False)

    t0 = time.time()
    res = ripser(wp_flat[sub], maxdim=1)   # direct 18D
    elapsed = time.time() - t0
    h0, h1 = res["dgms"]
    betti0 = count_persistent_clusters(h0, gap_ratio=1.5)
    h1_pers = h1[:, 1] - h1[:, 0]
    h1_f = h1_pers[np.isfinite(h1_pers)]
    h0_top = np.sort((h0[:, 1] - h0[:, 0])[np.isfinite(h0[:, 1] - h0[:, 0])])[::-1][:5]
    h1_top = np.sort(h1_f)[::-1][:5] if len(h1_f) else np.array([])

    print(f"  ripser(18D raw) in {elapsed:.1f}s")
    print(f"  B0 = {betti0}")
    print(f"  H0 top5 persistence: {h0_top.round(3)}")
    print(f"  H1 top5 persistence: {h1_top.round(3)}")
    print(f"  H1 loops >0.3: {(h1_f > 0.3).sum() if len(h1_f) else 0}")

    return {
        "label": label,
        "wp_3d": wp_3d,
        "wp_flat": wp_flat,
        "X_pca": X_pca,
        "pca_components": pca.components_,
        "var_exp": pca.explained_variance_ratio_,
        "diagrams": res["dgms"],
        "betti_0": betti0,
        "h0_top": h0_top,
        "h1_top": h1_top,
        "id_raw": id_raw,
        "id_pca": id_pca,
    }


def cluster_assignment(wp_flat, k):
    """K-means assign each trajectory to a mode cluster."""
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(wp_flat)
    return km.labels_, km.cluster_centers_


def mode_summary(res, k):
    """Assign trajectories to k clusters and report per-mode stats."""
    labels, centers = cluster_assignment(res["wp_flat"], k)
    print(f"\n  Mode assignment (k={k}):")
    for m in range(k):
        mask = labels == m
        count = mask.sum()
        wp = res["wp_3d"][mask]
        mean_wp = wp.mean(axis=0)
        print(f"    mode {m}: N={count}  ({100*count/len(labels):.1f}%)")
        print(f"      waypoint means:")
        for w_idx in range(3):
            arr = mean_wp[w_idx]
            print(f"        w{w_idx+1}: [{', '.join(f'{x:+.2f}' for x in arr)}]")
    return labels, centers


# ------------------------- plots -------------------------

def plot_wp_pca_2d(res_o, res_f, save_path, labels_o=None, labels_f=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, res, labels, color in zip(
        axes,
        [res_o, res_f],
        [labels_o, labels_f],
        ["tab:red", "tab:blue"],
    ):
        X = res["X_pca"][:, :2]
        if labels is not None:
            cmap = plt.cm.get_cmap("tab10")
            for m in np.unique(labels):
                mk = labels == m
                ax.scatter(X[mk, 0], X[mk, 1], s=4,
                           color=cmap(m), alpha=0.5,
                           label=f"mode {m} (N={mk.sum()})")
            ax.legend(fontsize=8, loc="best")
        else:
            ax.scatter(X[:, 0], X[:, 1], s=3, color=color, alpha=0.35)
        ax.set_xlabel("PC1 (waypoint-space)")
        ax.set_ylabel("PC2 (waypoint-space)")
        ax.set_title(f"{res['label']}  B0={res['betti_0']}  ID={res['id_pca']:.2f}")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Waypoint-space (18D → 2D PCA). Each point = one trajectory's 3 waypoints", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_wp_pca_3d(res_o, res_f, save_path, labels_o=None, labels_f=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(14, 6))
    for i, (res, labels, color) in enumerate(zip(
        [res_o, res_f], [labels_o, labels_f], ["tab:red", "tab:blue"])):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        X = res["X_pca"][:, :3]
        if labels is not None:
            cmap = plt.cm.get_cmap("tab10")
            for m in np.unique(labels):
                mk = labels == m
                ax.scatter(X[mk, 0], X[mk, 1], X[mk, 2], s=3,
                           color=cmap(m), alpha=0.4, label=f"mode {m}")
            ax.legend(fontsize=8)
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, color=color, alpha=0.35)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(f"{res['label']}  B0={res['betti_0']}  ID={res['id_pca']:.2f}")
    plt.suptitle("Waypoint-space 3D (each point = one trajectory)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_persistence(res_o, res_f, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res in zip(axes, [res_o, res_f]):
        plot_diagrams(res["diagrams"], ax=ax, show=False)
        ax.set_title(f"{res['label']}: B0={res['betti_0']}, ID(18D)={res['id_raw']:.2f}")
    plt.suptitle("Waypoint-space Persistence Diagrams (direct 18D)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_mode_waypoints_3d(res, labels, save_path):
    """For collision-free: show each mode's 3 waypoints in joint space 3D (PC1-3)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    k = len(np.unique(labels))
    fig = plt.figure(figsize=(5 * k, 5))
    cmap = plt.cm.get_cmap("tab10")

    # PCA over all waypoints combined
    all_wp = res["wp_3d"].reshape(-1, 6)   # [N*3, 6]
    pca = PCA(n_components=3).fit(all_wp)

    for m in range(k):
        ax = fig.add_subplot(1, k, m + 1, projection='3d')
        mask = labels == m
        wp_m = res["wp_3d"][mask]            # [n_m, 3, 6]

        # Plot each trajectory's 3 waypoints as connected line
        for wp_seq in wp_m[::20]:            # subsample for clarity
            pts = pca.transform(wp_seq)      # [3, 3]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=cmap(m), alpha=0.25, linewidth=0.8)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       color=cmap(m), s=8, alpha=0.6)

        # Highlight mean waypoint trajectory
        mean_wp = wp_m.mean(axis=0)          # [3, 6]
        mean_pts = pca.transform(mean_wp)
        ax.plot(mean_pts[:, 0], mean_pts[:, 1], mean_pts[:, 2],
                color='black', linewidth=3, alpha=0.9,
                label=f"mean (N={mask.sum()})")
        ax.scatter(mean_pts[:, 0], mean_pts[:, 1], mean_pts[:, 2],
                   color='black', s=80, marker='*', zorder=5)
        for w_idx, (x, y, z) in enumerate(mean_pts):
            ax.text(x, y, z, f" w{w_idx+1}", fontsize=10, color='black')

        ax.set_xlabel("joint-PC1"); ax.set_ylabel("joint-PC2"); ax.set_zlabel("joint-PC3")
        ax.set_title(f"Mode {m}  (N={mask.sum()})")
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle(f"{res['label']}: Waypoint sequences per mode (3D joint-PCA)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_waypoint_distribution(res_o, res_f, save_path):
    """For each of the 3 waypoints, compare joint distribution Original vs Col-Free."""
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharey='row')
    for w in range(3):
        for j in range(6):
            ax = axes[w, j]
            vo = res_o["wp_3d"][:, w, j]
            vf = res_f["wp_3d"][:, w, j]
            ax.hist(vo, bins=40, alpha=0.5, color='tab:red', label='Orig', density=True)
            ax.hist(vf, bins=40, alpha=0.5, color='tab:blue', label='Free', density=True)
            if w == 0:
                ax.set_title(f"joint {j}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"w{w+1}", fontsize=10)
            if w == 0 and j == 0:
                ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
    plt.suptitle("Per-waypoint × per-joint distribution (red=Orig, blue=Col-Free)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_h0_gap(res_o, res_f, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, res, color in zip(axes, [res_o, res_f], ["tab:red", "tab:blue"]):
        h0 = res["diagrams"][0]
        pers = h0[:, 1] - h0[:, 0]
        top = np.sort(pers[np.isfinite(pers)])[::-1][:12]
        ax.bar(range(1, len(top) + 1), top, color=color, alpha=0.75)
        ax.set_xlabel("rank"); ax.set_ylabel("H0 persistence")
        ax.set_title(f"{res['label']}: B0={res['betti_0']}")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Top-12 H0 persistence (waypoint-space 18D)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ------------------------- main -------------------------

def main():
    print("=== Topology Analysis v3: waypoint-based ===", flush=True)

    d_o = np.load(os.path.join(ROOT, "outputs/results/trajectory_dataset/trajectories_5000x100x6.npy"))
    d_f = np.load(os.path.join(ROOT, "outputs/results/collision_free_dataset/trajectories_collision_free_5000x100x6.npy"))

    # Verify waypoint extraction (at segment boundary, vel should be ~0)
    wp, _ = extract_waypoints(d_o)
    print(f"\nOriginal shape: {d_o.shape}")
    print(f"Extracted waypoints shape: {wp.shape}  (at indices {WP_INDICES})")

    res_o = analyze_waypoint_space(d_o, "Original")
    res_f = analyze_waypoint_space(d_f, "Collision-Free")

    # Mode clustering
    print(f"\n--- Mode clustering (Original, k={res_o['betti_0']}) ---")
    k_o = max(res_o["betti_0"], 1)
    labels_o, _ = mode_summary(res_o, k_o)

    print(f"\n--- Mode clustering (Collision-Free, k={res_f['betti_0']}) ---")
    k_f = max(res_f["betti_0"], 1)
    labels_f, _ = mode_summary(res_f, k_f)

    # Plots
    plot_wp_pca_2d(res_o, res_f, os.path.join(SAVE, "wp_pca_2d.png"),
                   labels_o=labels_o, labels_f=labels_f)
    plot_wp_pca_3d(res_o, res_f, os.path.join(SAVE, "wp_pca_3d.png"),
                   labels_o=labels_o, labels_f=labels_f)
    plot_persistence(res_o, res_f, os.path.join(SAVE, "wp_persistence_diagrams.png"))
    plot_h0_gap(res_o, res_f, os.path.join(SAVE, "wp_h0_gap.png"))
    plot_per_waypoint_distribution(res_o, res_f,
                                    os.path.join(SAVE, "wp_joint_distribution.png"))
    plot_mode_waypoints_3d(res_f, labels_f,
                            os.path.join(SAVE, "wp_mode_sequences_free.png"))
    if k_o > 1:
        plot_mode_waypoints_3d(res_o, labels_o,
                                os.path.join(SAVE, "wp_mode_sequences_orig.png"))

    # Summary
    print(f"\n{'='*72}")
    print("WAYPOINT-SPACE TDA SUMMARY")
    print(f"{'='*72}")
    print(f"{'Metric':<36} {'Original':>15} {'Collision-Free':>18}")
    print("-" * 72)
    print(f"{'B0 (# solution families)':<36} "
          f"{res_o['betti_0']:>15d} {res_f['betti_0']:>18d}")
    print(f"{'ID (raw 18D)':<36} "
          f"{res_o['id_raw']:>15.2f} {res_f['id_raw']:>18.2f}")
    print(f"{'ID (10D PCA)':<36} "
          f"{res_o['id_pca']:>15.2f} {res_f['id_pca']:>18.2f}")
    print(f"{'H0 top persistence':<36} "
          f"{res_o['h0_top'][0]:>15.3f} {res_f['h0_top'][0]:>18.3f}")
    print(f"{'H0 gap (top1/top2)':<36} "
          f"{res_o['h0_top'][0]/res_o['h0_top'][1]:>15.2f} "
          f"{res_f['h0_top'][0]/res_f['h0_top'][1]:>18.2f}")

    np.save(os.path.join(SAVE, "results_v3.npy"), {
        "orig": {k: v for k, v in res_o.items()
                 if k not in ["diagrams", "wp_3d", "wp_flat", "X_pca", "pca_components"]},
        "free": {k: v for k, v in res_f.items()
                 if k not in ["diagrams", "wp_3d", "wp_flat", "X_pca", "pca_components"]},
        "labels_o": labels_o, "labels_f": labels_f,
    })
    print(f"\n=== Done. Output: {SAVE} ===")


if __name__ == "__main__":
    main()
