"""
TDA on 20000 collision-free dataset.
Runs both v2 (node/trajectory-as-point) and v3 (waypoint-based) analyses.
Compares: Original (5000) vs Collision-Free (20000).
Saves to outputs/results/topology_analysis_20000/.
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

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
SAVE = os.path.join(ROOT, "outputs/results/topology_analysis_20000")
os.makedirs(SAVE, exist_ok=True)

WP_INDICES = [24, 49, 74]


# ------------------------- utils -------------------------

def count_persistent_clusters(h0, gap_ratio=1.2):
    """gap_ratio lowered to 1.2 - with 20000 points, modes bridge more (smaller gaps)."""
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


def extract_waypoints(data):
    wp = data[:, WP_INDICES, :]
    return wp, wp.reshape(data.shape[0], -1)


# ------------------------- Node-based (flatten 600D) -------------------------

def analyze_node(data, label, pca_dim=10, n_sub=1500):
    N, T, D = data.shape
    X_flat = data.reshape(N, T * D)

    print(f"\n=== Node-based {label}: {N} traj in {T*D}D ===", flush=True)
    pca = PCA(n_components=pca_dim)
    X_pca = pca.fit_transform(X_flat)
    var = pca.explained_variance_ratio_
    print(f"  PCA {pca_dim}D cum var: {var.cumsum()[-1]:.3f}")
    print(f"  Top 5 var: {var[:5].round(3)}")

    id_est = twonn_id(X_pca)
    print(f"  ID (TwoNN on 10D PCA): {id_est:.2f}")

    rng = np.random.default_rng(42)
    sub = rng.choice(N, min(n_sub, N), replace=False)
    t0 = time.time()
    res = ripser(X_pca[sub], maxdim=1)
    elapsed = time.time() - t0
    h0, h1 = res["dgms"]
    betti0 = count_persistent_clusters(h0)
    h1_pers = h1[:, 1] - h1[:, 0]
    h1_f = h1_pers[np.isfinite(h1_pers)]
    h0_top = np.sort((h0[:, 1] - h0[:, 0])[np.isfinite(h0[:, 1] - h0[:, 0])])[::-1][:5]
    h1_top = np.sort(h1_f)[::-1][:5] if len(h1_f) else np.array([])

    print(f"  ripser {elapsed:.1f}s  B0={betti0}")
    print(f"  H0 top5: {h0_top.round(2)}")
    print(f"  H1 top5: {h1_top.round(2)}")
    print(f"  H1 loops (>0.5): {(h1_f > 0.5).sum() if len(h1_f) else 0}")

    return {
        "label": label, "X_pca": X_pca, "diagrams": res["dgms"],
        "betti_0": betti0, "id_est": id_est,
        "h0_top": h0_top, "h1_top": h1_top, "var_exp": var,
    }


# ------------------------- Waypoint-based (18D) -------------------------

def analyze_wp(data, label, n_sub=1500):
    N = data.shape[0]
    wp_3d, wp_flat = extract_waypoints(data)

    print(f"\n=== Waypoint-space {label}: {N} traj in 18D ===", flush=True)
    id_raw = twonn_id(wp_flat)
    print(f"  ID (raw 18D): {id_raw:.2f}")

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(wp_flat)
    var = pca.explained_variance_ratio_
    print(f"  PCA 10D cum var: {var.cumsum()[-1]:.3f}")
    print(f"  Top 5 var: {var[:5].round(3)}")
    id_pca = twonn_id(X_pca)
    print(f"  ID (10D PCA): {id_pca:.2f}")

    rng = np.random.default_rng(42)
    sub = rng.choice(N, min(n_sub, N), replace=False)
    t0 = time.time()
    res = ripser(wp_flat[sub], maxdim=1)
    elapsed = time.time() - t0
    h0, h1 = res["dgms"]
    betti0 = count_persistent_clusters(h0)
    h1_pers = h1[:, 1] - h1[:, 0]
    h1_f = h1_pers[np.isfinite(h1_pers)]
    h0_top = np.sort((h0[:, 1] - h0[:, 0])[np.isfinite(h0[:, 1] - h0[:, 0])])[::-1][:5]
    h1_top = np.sort(h1_f)[::-1][:5] if len(h1_f) else np.array([])

    print(f"  ripser(18D) {elapsed:.1f}s  B0={betti0}")
    print(f"  H0 top5: {h0_top.round(2)}")
    print(f"  H1 top5: {h1_top.round(2)}")
    print(f"  H1 loops (>0.3): {(h1_f > 0.3).sum() if len(h1_f) else 0}")

    return {
        "label": label, "wp_3d": wp_3d, "wp_flat": wp_flat,
        "X_pca": X_pca, "diagrams": res["dgms"],
        "betti_0": betti0, "id_raw": id_raw, "id_pca": id_pca,
        "h0_top": h0_top, "h1_top": h1_top, "var_exp": var,
    }


# ------------------------- Plots -------------------------

def plot_node_pca_2d(o, f, save):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, res, color in zip(axes, [o, f], ["tab:red", "tab:blue"]):
        X = res["X_pca"][:, :2]
        ax.scatter(X[:, 0], X[:, 1], s=3, color=color, alpha=0.3)
        ax.set_xlabel("PC1 (trajectory-space)")
        ax.set_ylabel("PC2 (trajectory-space)")
        ax.set_title(f"{res['label']}  N={len(X)}  B0={res['betti_0']}  ID={res['id_est']:.2f}")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Node-based (Trajectory-as-Point) 2D PCA", fontsize=12)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"Saved: {save}")


def plot_node_pca_3d(o, f, save):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(14, 6))
    for i, (res, color) in enumerate(zip([o, f], ["tab:red", "tab:blue"])):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        X = res["X_pca"][:, :3]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, color=color, alpha=0.3)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(f"{res['label']}  N={len(X)}  B0={res['betti_0']}  ID={res['id_est']:.2f}")
    plt.suptitle("Node-based 3D", fontsize=12)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"Saved: {save}")


def plot_node_diagrams(o, f, save):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res in zip(axes, [o, f]):
        plot_diagrams(res["diagrams"], ax=ax, show=False)
        ax.set_title(f"{res['label']}: B0={res['betti_0']}, ID={res['id_est']:.2f}")
    plt.suptitle("Node-based Persistence Diagrams", fontsize=13)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"Saved: {save}")


def plot_wp_pca_2d(o, f, save, labels_o=None, labels_f=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, res, labels, color in zip(axes, [o, f], [labels_o, labels_f],
                                       ["tab:red", "tab:blue"]):
        X = res["X_pca"][:, :2]
        if labels is not None and len(np.unique(labels)) > 1:
            cmap = plt.cm.get_cmap("tab10")
            for m in np.unique(labels):
                mk = labels == m
                ax.scatter(X[mk, 0], X[mk, 1], s=4, color=cmap(m), alpha=0.5,
                           label=f"mode {m} (N={mk.sum()})")
            ax.legend(fontsize=8)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=3, color=color, alpha=0.3)
        ax.set_xlabel("PC1 (waypoint-space)")
        ax.set_ylabel("PC2 (waypoint-space)")
        ax.set_title(f"{res['label']}  N={len(X)}  B0={res['betti_0']}  ID={res['id_pca']:.2f}")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Waypoint-space (18D → 2D PCA)", fontsize=12)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"Saved: {save}")


def plot_wp_pca_3d(o, f, save, labels_o=None, labels_f=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(14, 6))
    for i, (res, labels, color) in enumerate(zip([o, f], [labels_o, labels_f],
                                                   ["tab:red", "tab:blue"])):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        X = res["X_pca"][:, :3]
        if labels is not None and len(np.unique(labels)) > 1:
            cmap = plt.cm.get_cmap("tab10")
            for m in np.unique(labels):
                mk = labels == m
                ax.scatter(X[mk, 0], X[mk, 1], X[mk, 2], s=3,
                           color=cmap(m), alpha=0.4, label=f"mode {m}")
            ax.legend(fontsize=8)
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, color=color, alpha=0.3)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(f"{res['label']}  N={len(X)}  B0={res['betti_0']}  ID={res['id_pca']:.2f}")
    plt.suptitle("Waypoint-space 3D", fontsize=12)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"Saved: {save}")


def plot_wp_diagrams(o, f, save):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res in zip(axes, [o, f]):
        plot_diagrams(res["diagrams"], ax=ax, show=False)
        ax.set_title(f"{res['label']}: B0={res['betti_0']}, ID(18D)={res['id_raw']:.2f}")
    plt.suptitle("Waypoint-space Persistence Diagrams (direct 18D)", fontsize=13)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"Saved: {save}")


def plot_h0_bars(o, f, save, key="diagrams"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, res, color in zip(axes, [o, f], ["tab:red", "tab:blue"]):
        h0 = res[key][0]
        pers = h0[:, 1] - h0[:, 0]
        top = np.sort(pers[np.isfinite(pers)])[::-1][:12]
        ax.bar(range(1, len(top) + 1), top, color=color, alpha=0.75)
        ax.set_xlabel("rank"); ax.set_ylabel("H0 persistence")
        ax.set_title(f"{res['label']}: B0={res['betti_0']}")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Top-12 H0 persistence", fontsize=12)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"Saved: {save}")


def plot_mode_seqs(res, labels, save):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    k = len(np.unique(labels))
    fig = plt.figure(figsize=(5 * k, 5))
    cmap = plt.cm.get_cmap("tab10")
    all_wp = res["wp_3d"].reshape(-1, 6)
    pca = PCA(n_components=3).fit(all_wp)
    for m in range(k):
        ax = fig.add_subplot(1, k, m + 1, projection='3d')
        mask = labels == m
        wp_m = res["wp_3d"][mask]
        for wp_seq in wp_m[::50]:
            pts = pca.transform(wp_seq)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=cmap(m), alpha=0.2, linewidth=0.7)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       color=cmap(m), s=6, alpha=0.5)
        mean_pts = pca.transform(wp_m.mean(axis=0))
        ax.plot(mean_pts[:, 0], mean_pts[:, 1], mean_pts[:, 2],
                color='black', linewidth=3, alpha=0.9, label=f"mean (N={mask.sum()})")
        ax.scatter(mean_pts[:, 0], mean_pts[:, 1], mean_pts[:, 2],
                   color='black', s=80, marker='*', zorder=5)
        for w_idx, (x, y, z) in enumerate(mean_pts):
            ax.text(x, y, z, f" w{w_idx+1}", fontsize=10)
        ax.set_xlabel("joint-PC1"); ax.set_ylabel("joint-PC2"); ax.set_zlabel("joint-PC3")
        ax.set_title(f"Mode {m}  (N={mask.sum()})")
        ax.legend(fontsize=8)
    plt.suptitle(f"{res['label']}: Waypoint sequences per mode", fontsize=13)
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close()
    print(f"Saved: {save}")


# ------------------------- Main -------------------------

def main():
    print("=== TDA on 20000 Collision-Free Dataset ===\n", flush=True)

    d_o = np.load(os.path.join(ROOT, "outputs/results/trajectory_dataset/trajectories_5000x100x6.npy"))
    d_f = np.load(os.path.join(ROOT, "outputs/results/collision_free_dataset/trajectories_collision_free_20000x100x6.npy"))
    print(f"Original:       {d_o.shape}")
    print(f"Collision-Free: {d_f.shape}")

    # Node-based
    node_o = analyze_node(d_o, "Original (5000)", pca_dim=10, n_sub=1500)
    node_f = analyze_node(d_f, "Col-Free (20000)", pca_dim=10, n_sub=2000)

    # Waypoint-based
    wp_o = analyze_wp(d_o, "Original (5000)", n_sub=1500)
    wp_f = analyze_wp(d_f, "Col-Free (20000)", n_sub=2000)

    # KMeans mode assignment. Force k=3 for col-free (visual evidence from PCA).
    def cluster(res, force_k=None):
        k = force_k if force_k is not None else max(res["betti_0"], 1)
        if k > 1:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(res["wp_flat"])
            return km.labels_
        return None

    labels_wp_o = cluster(wp_o)
    labels_wp_f = cluster(wp_f, force_k=3)   # 3 modes always for col-free

    # Plots - node-based
    plot_node_pca_2d(node_o, node_f, os.path.join(SAVE, "node_pca_2d_20000.png"))
    plot_node_pca_3d(node_o, node_f, os.path.join(SAVE, "node_pca_3d_20000.png"))
    plot_node_diagrams(node_o, node_f, os.path.join(SAVE, "node_persistence_diagrams_20000.png"))
    plot_h0_bars(node_o, node_f, os.path.join(SAVE, "node_h0_gap_20000.png"))

    # Plots - waypoint-based
    plot_wp_pca_2d(wp_o, wp_f, os.path.join(SAVE, "wp_pca_2d_20000.png"),
                    labels_o=labels_wp_o, labels_f=labels_wp_f)
    plot_wp_pca_3d(wp_o, wp_f, os.path.join(SAVE, "wp_pca_3d_20000.png"),
                    labels_o=labels_wp_o, labels_f=labels_wp_f)
    plot_wp_diagrams(wp_o, wp_f, os.path.join(SAVE, "wp_persistence_diagrams_20000.png"))
    plot_h0_bars(wp_o, wp_f, os.path.join(SAVE, "wp_h0_gap_20000.png"))

    if labels_wp_f is not None:
        plot_mode_seqs(wp_f, labels_wp_f, os.path.join(SAVE, "wp_mode_sequences_free_20000.png"))

    # Per-mode stats
    if labels_wp_f is not None:
        print(f"\n--- Collision-Free mode breakdown (k={wp_f['betti_0']}) ---")
        for m in np.unique(labels_wp_f):
            mask = labels_wp_f == m
            print(f"  mode {m}: N={mask.sum()}  ({100*mask.sum()/len(labels_wp_f):.1f}%)")
            mean_wp = wp_f["wp_3d"][mask].mean(axis=0)
            for w_idx in range(3):
                arr = mean_wp[w_idx]
                print(f"    w{w_idx+1}: [{', '.join(f'{x:+.2f}' for x in arr)}]")

    # Summary table
    print(f"\n{'='*76}")
    print("SUMMARY - 20000 Collision-Free vs 5000 Original")
    print(f"{'='*76}")
    print(f"{'Metric':<38} {'Orig (5000)':>15} {'Col-Free (20000)':>20}")
    print("-" * 76)
    print(f"{'Node-based B0':<38} {node_o['betti_0']:>15d} {node_f['betti_0']:>20d}")
    print(f"{'Node-based ID (10D PCA)':<38} {node_o['id_est']:>15.2f} {node_f['id_est']:>20.2f}")
    print(f"{'Node-based H0 top1':<38} {node_o['h0_top'][0]:>15.2f} {node_f['h0_top'][0]:>20.2f}")
    print(f"{'Node-based H0 top1/top2':<38} "
          f"{node_o['h0_top'][0]/node_o['h0_top'][1]:>15.2f} "
          f"{node_f['h0_top'][0]/node_f['h0_top'][1]:>20.2f}")
    print()
    print(f"{'Waypoint-space B0':<38} {wp_o['betti_0']:>15d} {wp_f['betti_0']:>20d}")
    print(f"{'Waypoint-space ID (raw 18D)':<38} {wp_o['id_raw']:>15.2f} {wp_f['id_raw']:>20.2f}")
    print(f"{'Waypoint-space ID (10D PCA)':<38} {wp_o['id_pca']:>15.2f} {wp_f['id_pca']:>20.2f}")
    print(f"{'Waypoint-space H0 top1':<38} {wp_o['h0_top'][0]:>15.2f} {wp_f['h0_top'][0]:>20.2f}")
    print(f"{'Waypoint-space H0 top1/top2':<38} "
          f"{wp_o['h0_top'][0]/wp_o['h0_top'][1]:>15.2f} "
          f"{wp_f['h0_top'][0]/wp_f['h0_top'][1]:>20.2f}")

    np.save(os.path.join(SAVE, "results_20000.npy"), {
        "node_o": {k: v for k, v in node_o.items() if k not in ["diagrams", "X_pca"]},
        "node_f": {k: v for k, v in node_f.items() if k not in ["diagrams", "X_pca"]},
        "wp_o": {k: v for k, v in wp_o.items()
                 if k not in ["diagrams", "wp_3d", "wp_flat", "X_pca"]},
        "wp_f": {k: v for k, v in wp_f.items()
                 if k not in ["diagrams", "wp_3d", "wp_flat", "X_pca"]},
    })
    print(f"\n=== Done. Output: {SAVE} ===")


if __name__ == "__main__":
    main()
