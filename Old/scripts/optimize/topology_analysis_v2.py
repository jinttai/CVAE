"""
Topology analysis v2 — fixed B0 counting + node-based (trajectory-as-point) TDA.

Fixes over v1:
  - B0 via persistence gap (elbow) instead of only infinite bars
  - Add per-scale cluster count for better multimodality detection

New: node-based TDA
  - Each full trajectory = one point in 600D (100 timesteps × 6 joints)
  - PCA-reduce to lower dim, then run persistent homology
  - Reveals topology of "trajectory manifold" (≠ per-timestep density manifold)
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

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
SAVE = os.path.join(ROOT, "outputs/results/topology_analysis_v2")
os.makedirs(SAVE, exist_ok=True)


# ------------------------- B0 proper counting -------------------------

def count_persistent_clusters(h0_diagram, gap_ratio=2.0):
    """
    Count meaningful clusters from H0 persistence diagram.

    Strategy: sort finite persistences descending. If top-k+1 persistence
    drops by factor >= gap_ratio from top-k, return k+1 clusters (the
    k largest gaps are "real").
    Infinite bar always present → +1.
    """
    pers = h0_diagram[:, 1] - h0_diagram[:, 0]
    finite = np.sort(pers[np.isfinite(pers)])[::-1]
    if len(finite) == 0:
        return 1
    # k = # finite bars that are much longer than the next one
    k_real = 0
    for i in range(min(20, len(finite) - 1)):
        ratio = finite[i] / max(finite[i + 1], 1e-12)
        if ratio >= gap_ratio:
            k_real = i + 1
    return k_real + 1   # + infinite bar


def cluster_count_at_scale(h0_diagram, eps):
    """# components alive at scale eps."""
    return int((h0_diagram[:, 1] > eps).sum())


def twonn_id(X, k=2):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nbrs.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    valid = (r1 > 1e-10) & (r2 > r1 + 1e-10)
    return np.log(2) / np.log(r2[valid] / r1[valid]).mean()


# ------------------------- Per-timestep (fixed) -------------------------

def analyze_per_timestep(data, label, n_sub=800, timesteps=(25, 50, 75)):
    N, T, D = data.shape
    rng = np.random.default_rng(42)
    sub = rng.choice(N, min(n_sub, N), replace=False)
    data_sub = data[sub]
    print(f"\n=== Per-timestep {label} ({n_sub}) ===", flush=True)

    out = {}
    for ts in timesteps:
        X = data_sub[:, ts, :]
        t0 = time.time()
        res = ripser(X, maxdim=1)
        elapsed = time.time() - t0
        dgms = res["dgms"]
        h0, h1 = dgms[0], dgms[1]

        betti0 = count_persistent_clusters(h0, gap_ratio=1.8)
        h1_pers = h1[:, 1] - h1[:, 0]
        h1_f = h1_pers[np.isfinite(h1_pers)]
        h1_long = int((h1_f > 0.2).sum()) if len(h1_f) else 0

        # top 5 H0 persistence (for gap visualization)
        h0_pers = h0[:, 1] - h0[:, 0]
        top5 = np.sort(h0_pers[np.isfinite(h0_pers)])[::-1][:5]

        out[ts] = {
            "diagrams": dgms,
            "betti_0": betti0,
            "h1_long": h1_long,
            "h1_max": h1_f.max() if len(h1_f) else 0.0,
            "h0_top5": top5,
            "time": elapsed,
        }
        print(f"  t={ts} ({ts*0.1:.1f}s): B0={betti0}, H1 long={h1_long}, "
              f"max H1={out[ts]['h1_max']:.3f}, time={elapsed:.1f}s")
        print(f"    H0 top5 persistence: {top5}")

    return out


# ------------------------- Node-based (trajectory-as-point) -------------------------

def analyze_node_based(data, label, pca_dim=10, n_sub=1500):
    """Each trajectory = 600D point. PCA-reduce, then TDA."""
    N, T, D = data.shape
    X_flat = data.reshape(N, T * D)  # [N, 600]

    print(f"\n=== Node-based {label}: {N} trajectories in {T*D}D ===", flush=True)

    pca = PCA(n_components=pca_dim)
    X_pca = pca.fit_transform(X_flat)
    var_exp = pca.explained_variance_ratio_
    print(f"  PCA({pca_dim}D) cumulative variance: {var_exp.cumsum()[-1]:.3f}")
    print(f"  Top 5 var: {var_exp[:5].round(3)}")

    # TwoNN intrinsic dim
    id_est = twonn_id(X_pca)
    print(f"  Intrinsic dim (TwoNN in {pca_dim}D PCA): {id_est:.2f}")

    # Subsample for ripser (expensive)
    rng = np.random.default_rng(42)
    sub = rng.choice(N, min(n_sub, N), replace=False)
    X_sub = X_pca[sub]

    t0 = time.time()
    res = ripser(X_sub, maxdim=1)
    elapsed = time.time() - t0
    dgms = res["dgms"]
    h0, h1 = dgms[0], dgms[1]

    betti0 = count_persistent_clusters(h0, gap_ratio=1.8)
    h1_pers = h1[:, 1] - h1[:, 0]
    h1_f = h1_pers[np.isfinite(h1_pers)]
    h1_top5 = np.sort(h1_f)[::-1][:5] if len(h1_f) else np.array([])
    h0_pers = h0[:, 1] - h0[:, 0]
    h0_top5 = np.sort(h0_pers[np.isfinite(h0_pers)])[::-1][:5]

    print(f"  ripser computed in {elapsed:.1f}s")
    print(f"  B0 (trajectory-space clusters): {betti0}")
    print(f"  H0 top5 persistence: {h0_top5}")
    print(f"  H1 loops (>0.5 pers): {(h1_f > 0.5).sum() if len(h1_f) else 0}")
    print(f"  H1 top5 persistence: {h1_top5}")

    return {
        "label": label,
        "X_pca": X_pca,
        "X_sub": X_sub,
        "diagrams": dgms,
        "betti_0": betti0,
        "h0_top5": h0_top5,
        "h1_top5": h1_top5,
        "id_est": id_est,
        "var_exp": var_exp,
        "pca_components": pca.components_,
    }


# ------------------------- Plots -------------------------

def plot_h0_gap_bars(res_orig, res_free, timesteps, save_path):
    """Bar chart of top-10 H0 persistence per timestep/dataset, to visualize gap."""
    fig, axes = plt.subplots(2, len(timesteps), figsize=(5 * len(timesteps), 8), sharey='row')

    for col, ts in enumerate(timesteps):
        for row, (res, lab, color) in enumerate([
            (res_orig, "Original", "tab:red"),
            (res_free, "Collision-Free", "tab:blue")
        ]):
            ax = axes[row, col]
            h0 = res[ts]["diagrams"][0]
            pers = h0[:, 1] - h0[:, 0]
            top = np.sort(pers[np.isfinite(pers)])[::-1][:10]
            ax.bar(range(1, len(top) + 1), top, color=color, alpha=0.7)
            ax.set_title(f"{lab}  t={ts*0.1:.1f}s  B0={res[ts]['betti_0']}")
            ax.set_xlabel("rank")
            if col == 0:
                ax.set_ylabel("H0 persistence")
            ax.grid(True, alpha=0.3)

    plt.suptitle("Top-10 H0 Persistence (gap = cluster boundary)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_node_diagrams(nodes_o, nodes_f, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res in zip(axes, [nodes_o, nodes_f]):
        plot_diagrams(res["diagrams"], ax=ax, show=False)
        ax.set_title(f"{res['label']}: B0={res['betti_0']}, ID={res['id_est']:.2f}")
    plt.suptitle("Node-based (Trajectory-as-Point) Persistence Diagrams", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_node_pca_2d(nodes_o, nodes_f, save_path):
    """Project trajectories to first 2 PCs in trajectory-space; scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res, color in zip(axes,
                              [nodes_o, nodes_f],
                              ["tab:red", "tab:blue"]):
        X = res["X_pca"][:, :2]
        ax.scatter(X[:, 0], X[:, 1], s=3, color=color, alpha=0.3)
        ax.set_xlabel("PC1 (trajectory-space)")
        ax.set_ylabel("PC2 (trajectory-space)")
        ax.set_title(f"{res['label']}  N={len(X)}  B0={res['betti_0']}")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Each point = one full trajectory (600D → 2D PCA)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_node_pca_3d(nodes_o, nodes_f, save_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(14, 6))
    for i, (res, color) in enumerate(zip([nodes_o, nodes_f], ["tab:red", "tab:blue"])):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        X = res["X_pca"][:, :3]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, color=color, alpha=0.35)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(f"{res['label']}  B0={res['betti_0']}  ID={res['id_est']:.2f}")
    plt.suptitle("Trajectory-space 3D scatter (each point = one trajectory)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_betti_evolution_fixed(data_orig, data_free, save_path):
    """Re-compute B0 and H1 long loops over all timesteps with FIXED B0."""
    T = data_orig.shape[1]
    ts_list = list(range(5, T - 5, 5))
    rng = np.random.default_rng(42)
    n_sub = 500

    def compute(data, label):
        sub = rng.choice(data.shape[0], n_sub, replace=False)
        ds = data[sub]
        b0, h1l = [], []
        for i, t in enumerate(ts_list):
            res = ripser(ds[:, t, :], maxdim=1)
            h0, h1 = res["dgms"][0], res["dgms"][1]
            b0.append(count_persistent_clusters(h0, gap_ratio=1.8))
            h1_pers = h1[:, 1] - h1[:, 0]
            h1_f = h1_pers[np.isfinite(h1_pers)]
            h1l.append(int((h1_f > 0.2).sum()) if len(h1_f) else 0)
            if (i + 1) % 5 == 0:
                print(f"  {label}: {i+1}/{len(ts_list)}", flush=True)
        return b0, h1l

    b0_o, h1_o = compute(data_orig, "Original")
    b0_f, h1_f = compute(data_free, "Col-Free")
    t_arr = np.array(ts_list) * 0.1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(t_arr, b0_o, 'o-', label='Original', color='tab:red')
    ax.plot(t_arr, b0_f, 's-', label='Collision-Free', color='tab:blue')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("B0 (gap-based)")
    ax.set_title("B0 evolution (persistence-gap method)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_arr, h1_o, 'o-', label='Original', color='tab:red')
    ax.plot(t_arr, h1_f, 's-', label='Collision-Free', color='tab:blue')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("# H1 long loops (>0.2)")
    ax.set_title("H1 loops evolution")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Fixed Betti Evolution", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

    return {"t": t_arr, "b0_o": b0_o, "b0_f": b0_f,
            "h1_o": h1_o, "h1_f": h1_f}


# ------------------------- Main -------------------------

def main():
    print("=== Topology Analysis v2 ===", flush=True)
    d_orig = np.load(os.path.join(ROOT, "outputs/results/trajectory_dataset/trajectories_5000x100x6.npy"))
    d_free = np.load(os.path.join(ROOT, "outputs/results/collision_free_dataset/trajectories_collision_free_5000x100x6.npy"))
    print(f"Original shape:       {d_orig.shape}")
    print(f"Collision-Free shape: {d_free.shape}")

    TS = [25, 50, 75]

    # 1) Per-timestep with fixed B0
    res_o = analyze_per_timestep(d_orig, "Original", timesteps=TS)
    res_f = analyze_per_timestep(d_free, "Collision-Free", timesteps=TS)

    # 2) Node-based (trajectory-as-point)
    nodes_o = analyze_node_based(d_orig, "Original", pca_dim=10, n_sub=1500)
    nodes_f = analyze_node_based(d_free, "Collision-Free", pca_dim=10, n_sub=1500)

    # 3) Plots
    plot_h0_gap_bars(res_o, res_f, TS, os.path.join(SAVE, "h0_gap_bars.png"))
    plot_node_diagrams(nodes_o, nodes_f, os.path.join(SAVE, "node_persistence_diagrams.png"))
    plot_node_pca_2d(nodes_o, nodes_f, os.path.join(SAVE, "node_pca_2d.png"))
    plot_node_pca_3d(nodes_o, nodes_f, os.path.join(SAVE, "node_pca_3d.png"))

    betti_evo = plot_betti_evolution_fixed(d_orig, d_free,
                                            os.path.join(SAVE, "betti_evolution_fixed.png"))

    # 4) Summary
    print(f"\n{'='*72}")
    print("SUMMARY (v2)")
    print(f"{'='*72}")
    print(f"{'Metric':<38} {'Original':>15} {'Col-Free':>15}")
    print("-" * 72)
    for ts in TS:
        print(f"t={ts*0.1:.1f}s B0 (gap-based):              "
              f"{res_o[ts]['betti_0']:>15d} {res_f[ts]['betti_0']:>15d}")
        print(f"t={ts*0.1:.1f}s H1 long loops (>0.2):        "
              f"{res_o[ts]['h1_long']:>15d} {res_f[ts]['h1_long']:>15d}")
    print(f"\n-- Node-based (trajectory-as-point) --")
    print(f"{'Trajectory-space B0':<38} {nodes_o['betti_0']:>15d} {nodes_f['betti_0']:>15d}")
    print(f"{'Trajectory-space ID (10D PCA)':<38} "
          f"{nodes_o['id_est']:>15.2f} {nodes_f['id_est']:>15.2f}")
    print(f"{'H0 top persistence':<38} "
          f"{nodes_o['h0_top5'][0]:>15.3f} {nodes_f['h0_top5'][0]:>15.3f}")

    np.save(os.path.join(SAVE, "results_v2.npy"), {
        "per_ts_orig": {k: {kk: vv for kk, vv in v.items() if kk != 'diagrams'}
                         for k, v in res_o.items()},
        "per_ts_free": {k: {kk: vv for kk, vv in v.items() if kk != 'diagrams'}
                         for k, v in res_f.items()},
        "nodes_orig": {k: v for k, v in nodes_o.items()
                       if k not in ['diagrams', 'X_sub']},
        "nodes_free": {k: v for k, v in nodes_f.items()
                       if k not in ['diagrams', 'X_sub']},
        "betti_evo": betti_evo,
    })
    print(f"\n=== Done. Output: {SAVE} ===")


if __name__ == "__main__":
    main()
