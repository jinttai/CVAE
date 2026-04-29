"""
Topology analysis on PCA-embedded trajectory dataset.

Analyses:
  1. Intrinsic dimension estimation per timestep
  2. Persistent homology (H0, H1) at key timesteps
  3. Betti number evolution over time
  4. Original vs Collision-Free comparison
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time
from ripser import ripser
from persim import plot_diagrams, bottleneck
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
SAVE = os.path.join(ROOT, "outputs/results/topology_analysis")
os.makedirs(SAVE, exist_ok=True)


def twonn_id(X, k=2):
    """TwoNN intrinsic dimension estimator."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nbrs.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    valid = (r1 > 1e-10) & (r2 > r1 + 1e-10)
    return np.log(2) / np.log(r2[valid] / r1[valid]).mean()


def compute_betti_numbers(diagrams, threshold=None):
    """Count persistent features above threshold (infinite bars or long bars)."""
    betti = []
    for dim_d in diagrams:
        if dim_d.shape[0] == 0:
            betti.append(0)
            continue
        persistences = dim_d[:, 1] - dim_d[:, 0]
        if threshold is None:
            # Use max finite persistence as reference
            finite_p = persistences[np.isfinite(persistences)]
            if len(finite_p) == 0:
                betti.append(len(dim_d))
                continue
            thr = np.percentile(finite_p, 90) if len(finite_p) > 5 else 0.1
        else:
            thr = threshold
        betti.append(int((persistences > thr).sum()))
    return betti


def analyze_dataset(data, label, n_sub=800, timesteps=(25, 50, 75)):
    """
    data: [N, T, D] trajectory dataset
    Returns: dict with analysis results
    """
    N, T, D = data.shape
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(N, min(n_sub, N), replace=False)
    data_sub = data[sub_idx]  # [n_sub, T, D]

    results = {"label": label, "timesteps": timesteps, "per_t": {}}

    print(f"\n=== {label} ({n_sub} subsampled) ===", flush=True)

    # Intrinsic dim over all timesteps
    id_all = []
    for t in range(1, T - 1):  # skip t=0, T-1 (degenerate)
        try:
            id_all.append(twonn_id(data_sub[:, t, :]))
        except Exception:
            id_all.append(np.nan)
    results["id_timeseries"] = np.array(id_all)
    print(f"  Intrinsic dim: mean={np.nanmean(id_all):.2f}  "
          f"range=[{np.nanmin(id_all):.2f}, {np.nanmax(id_all):.2f}]")

    # Persistent homology at key timesteps
    for ts in timesteps:
        X = data_sub[:, ts, :]  # [n_sub, D]
        t0 = time.time()
        res = ripser(X, maxdim=1)  # H0, H1
        elapsed = time.time() - t0
        diagrams = res["dgms"]

        n_h0 = len(diagrams[0])
        n_h1 = len(diagrams[1])

        # Persistence = death - birth
        h0_pers = diagrams[0][:, 1] - diagrams[0][:, 0]
        h1_pers = diagrams[1][:, 1] - diagrams[1][:, 0]
        h0_finite = h0_pers[np.isfinite(h0_pers)]
        h1_finite = h1_pers[np.isfinite(h1_pers)]

        # Betti numbers at different thresholds
        betti_at_max_h0 = (np.isinf(h0_pers)).sum()  # # connected components
        h1_long = (h1_finite > 0.2).sum() if len(h1_finite) > 0 else 0
        h1_medium = (h1_finite > 0.1).sum() if len(h1_finite) > 0 else 0

        results["per_t"][ts] = {
            "diagrams": diagrams,
            "n_h0": n_h0,
            "n_h1": n_h1,
            "betti_0": betti_at_max_h0,
            "h1_long": h1_long,
            "h1_medium": h1_medium,
            "h0_max_persistence": h0_finite.max() if len(h0_finite) > 0 else 0,
            "h1_max_persistence": h1_finite.max() if len(h1_finite) > 0 else 0,
            "compute_time": elapsed,
        }

        print(f"\n  t={ts}: computed in {elapsed:.1f}s")
        print(f"    H0 features: {n_h0}  (Betti_0 = # components = {betti_at_max_h0})")
        print(f"    H1 features: {n_h1}  (long loops > 0.2 persistence: {h1_long})")
        print(f"    H0 max persistence: {h0_finite.max() if len(h0_finite) > 0 else 0:.4f}")
        print(f"    H1 max persistence: {h1_finite.max() if len(h1_finite) > 0 else 0:.4f}")

    return results


def plot_persistence_diagrams(results_orig, results_free, timesteps, save_path):
    """Side-by-side persistence diagrams for each timestep."""
    n_ts = len(timesteps)
    fig, axes = plt.subplots(2, n_ts, figsize=(5 * n_ts, 10))

    for col, ts in enumerate(timesteps):
        for row, results in enumerate([results_orig, results_free]):
            ax = axes[row, col]
            dgms = results["per_t"][ts]["diagrams"]
            plot_diagrams(dgms, ax=ax, show=False, legend=(row == 0 and col == 0))
            ax.set_title(f"{results['label']} — t={ts*0.1:.1f}s")

    plt.suptitle("Persistence Diagrams (H0: blue, H1: orange)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")


def plot_betti_evolution(results_orig, results_free, T, save_path, data_orig, data_free):
    """Compute Betti numbers across all timesteps for evolution plot."""
    rng = np.random.default_rng(42)
    n_sub = 500

    # For efficiency, compute only at sparse timesteps
    timesteps_sparse = list(range(5, T - 5, 5))  # every 5 steps
    print(f"\n=== Betti Evolution ({len(timesteps_sparse)} timesteps) ===", flush=True)

    def compute_over_time(data, label):
        sub_idx = rng.choice(data.shape[0], n_sub, replace=False)
        data_sub = data[sub_idx]
        h0_counts, h1_long_counts, h1_med_counts = [], [], []
        for i, ts in enumerate(timesteps_sparse):
            X = data_sub[:, ts, :]
            res = ripser(X, maxdim=1)
            h0_pers = res["dgms"][0][:, 1] - res["dgms"][0][:, 0]
            h1_pers = res["dgms"][1][:, 1] - res["dgms"][1][:, 0]
            h1_f = h1_pers[np.isfinite(h1_pers)]
            h0_counts.append(np.isinf(h0_pers).sum())
            h1_long_counts.append((h1_f > 0.2).sum() if len(h1_f) > 0 else 0)
            h1_med_counts.append((h1_f > 0.1).sum() if len(h1_f) > 0 else 0)
            if (i + 1) % 5 == 0:
                print(f"  {label}: {i+1}/{len(timesteps_sparse)}", flush=True)
        return h0_counts, h1_long_counts, h1_med_counts

    h0_o, h1l_o, h1m_o = compute_over_time(data_orig, "Original")
    h0_f, h1l_f, h1m_f = compute_over_time(data_free, "Col-Free")

    t_arr = np.array(timesteps_sparse) * 0.1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(t_arr, h0_o, 'o-', label='Original', color='tab:red')
    ax.plot(t_arr, h0_f, 's-', label='Collision-Free', color='tab:blue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Betti_0 (# components)")
    ax.set_title("H0: Connected Components over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_arr, h1m_o, 'o-', label='Orig H1 (>0.1)', color='tab:red', alpha=0.7)
    ax.plot(t_arr, h1m_f, 's-', label='Free H1 (>0.1)', color='tab:blue', alpha=0.7)
    ax.plot(t_arr, h1l_o, 'o--', label='Orig H1 (>0.2)', color='darkred')
    ax.plot(t_arr, h1l_f, 's--', label='Free H1 (>0.2)', color='darkblue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("# H1 features")
    ax.set_title("H1: Loops/Holes over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Betti Number Evolution: Original vs Collision-Free", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    return {"t_arr": t_arr, "h0_o": h0_o, "h0_f": h0_f,
            "h1l_o": h1l_o, "h1l_f": h1l_f, "h1m_o": h1m_o, "h1m_f": h1m_f}


def plot_id_evolution(id_o, id_f, T, save_path):
    """Intrinsic dimension evolution."""
    t_arr = np.arange(1, T - 1) * 0.1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_arr, id_o, '-', label='Original', color='tab:red', alpha=0.8)
    ax.plot(t_arr, id_f, '-', label='Collision-Free', color='tab:blue', alpha=0.8)
    ax.axhline(6, color='gray', linestyle='--', alpha=0.5, label='Ambient (6D)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intrinsic Dimension (TwoNN)")
    ax.set_title("Intrinsic Dimension Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=== Topology Analysis on Trajectory Dataset ===\n", flush=True)

    # Load data
    d_orig = np.load(os.path.join(ROOT, "outputs/results/trajectory_dataset/trajectories_5000x100x6.npy"))
    d_free = np.load(os.path.join(ROOT, "outputs/results/collision_free_dataset/trajectories_collision_free_5000x100x6.npy"))

    print(f"Original:       {d_orig.shape}")
    print(f"Collision-Free: {d_free.shape}")

    T = d_orig.shape[1]
    TIMESTEPS = [25, 50, 75]  # 2.5s, 5s, 7.5s

    # ---- Main analysis ----
    res_orig = analyze_dataset(d_orig, "Original", n_sub=800, timesteps=TIMESTEPS)
    res_free = analyze_dataset(d_free, "Collision-Free", n_sub=800, timesteps=TIMESTEPS)

    # ---- Plots ----
    plot_persistence_diagrams(res_orig, res_free, TIMESTEPS,
                               os.path.join(SAVE, "persistence_diagrams.png"))

    plot_id_evolution(res_orig["id_timeseries"], res_free["id_timeseries"], T,
                       os.path.join(SAVE, "intrinsic_dim_evolution.png"))

    betti_data = plot_betti_evolution(res_orig, res_free, T,
                                        os.path.join(SAVE, "betti_evolution.png"),
                                        d_orig, d_free)

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print("SUMMARY: Original vs Collision-Free")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Original':>15} {'Collision-Free':>18}")
    print("-" * 70)
    print(f"{'Intrinsic dim (mean)':<30} "
          f"{np.nanmean(res_orig['id_timeseries']):>15.2f} "
          f"{np.nanmean(res_free['id_timeseries']):>18.2f}")

    for ts in TIMESTEPS:
        ro = res_orig["per_t"][ts]
        rf = res_free["per_t"][ts]
        print(f"\n-- t={ts*0.1:.1f}s --")
        print(f"{'  Betti_0 (# components)':<30} {ro['betti_0']:>15d} {rf['betti_0']:>18d}")
        print(f"{'  H1 loops (>0.1)':<30} {ro['h1_medium']:>15d} {rf['h1_medium']:>18d}")
        print(f"{'  H1 long loops (>0.2)':<30} {ro['h1_long']:>15d} {rf['h1_long']:>18d}")
        print(f"{'  H1 max persistence':<30} {ro['h1_max_persistence']:>15.4f} "
              f"{rf['h1_max_persistence']:>18.4f}")

    # Save summary
    np.save(os.path.join(SAVE, "topology_results.npy"), {
        "results_orig": {k: v for k, v in res_orig.items() if k != "per_t"},
        "results_free": {k: v for k, v in res_free.items() if k != "per_t"},
        "betti_evolution": betti_data,
    })
    print(f"\n=== Done. Results in {SAVE} ===")


if __name__ == "__main__":
    main()
