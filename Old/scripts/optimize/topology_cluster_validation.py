"""
Validate whether the 3-mode structure in collision-free is real or artifact.

Tests:
  1. Silhouette score across k=2..8 (does k=3 stand out?)
  2. Gap statistic (compare to uniform null)
  3. DBSCAN (density-based, no k required)
  4. KDE valley check in 2D PCA (are there real density gaps?)
  5. Baseline: same tests on gaussian-sampled data of same shape
  6. Sensitivity: subsample 20000 down to 5000, re-measure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from ripser import ripser

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
SAVE = os.path.join(ROOT, "outputs/results/topology_cluster_validation")
os.makedirs(SAVE, exist_ok=True)

WP_INDICES = [24, 49, 74]


def extract_waypoints(data):
    return data[:, WP_INDICES, :].reshape(data.shape[0], -1)


# ==================== Test 1: Silhouette across k ====================

def silhouette_sweep(X, k_range=range(2, 9), n_sub=3000):
    rng = np.random.default_rng(42)
    if len(X) > n_sub:
        X = X[rng.choice(len(X), n_sub, replace=False)]
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
        s = silhouette_score(X, km.labels_, sample_size=2000, random_state=42)
        scores[k] = s
        print(f"  k={k}: silhouette={s:.4f}")
    return scores


# ==================== Test 2: Gap statistic (vs uniform null) ====================

def gap_statistic(X, k_range=range(1, 9), n_refs=5, n_sub=2000):
    rng = np.random.default_rng(42)
    if len(X) > n_sub:
        X = X[rng.choice(len(X), n_sub, replace=False)]

    # Wk = sum of intra-cluster distances for k clusters
    def Wk(data, k):
        km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(data)
        return km.inertia_

    logWks, gaps = [], []
    lo, hi = X.min(axis=0), X.max(axis=0)
    for k in k_range:
        Wk_data = Wk(X, k)
        # Reference distribution: uniform in bounding box
        logWk_refs = []
        for _ in range(n_refs):
            X_ref = rng.uniform(lo, hi, size=X.shape)
            logWk_refs.append(np.log(Wk(X_ref, k) + 1e-12))
        logWk_data = np.log(Wk_data + 1e-12)
        gap = np.mean(logWk_refs) - logWk_data
        logWks.append(logWk_data)
        gaps.append(gap)
        print(f"  k={k}: log(Wk)={logWk_data:.3f}  gap={gap:.3f}")
    return list(k_range), gaps, logWks


# ==================== Test 3: DBSCAN (no k required) ====================

def dbscan_scan(X, eps_list=None, min_samples=20, n_sub=3000):
    rng = np.random.default_rng(42)
    if len(X) > n_sub:
        X = X[rng.choice(len(X), n_sub, replace=False)]
    if eps_list is None:
        # Guess from 4th-nearest neighbor distance distribution
        from sklearn.neighbors import NearestNeighbors
        nb = NearestNeighbors(n_neighbors=min_samples).fit(X)
        d, _ = nb.kneighbors(X)
        q = np.percentile(d[:, -1], [50, 75, 90])
        eps_list = list(q)
        print(f"  suggested eps: {[f'{e:.3f}' for e in eps_list]}")

    results = []
    for eps in eps_list:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        # Cluster sizes (excluding noise)
        sizes = [(labels == c).sum() for c in set(labels) if c != -1]
        sizes.sort(reverse=True)
        results.append({
            "eps": eps, "n_clusters": n_clusters, "n_noise": n_noise,
            "top_sizes": sizes[:5],
        })
        print(f"  eps={eps:.3f}: n_clusters={n_clusters}  noise={n_noise}/{len(X)}  "
              f"top sizes={sizes[:5]}")
    return results


# ==================== Test 4: KDE valley in 2D ====================

def kde_valley_check(X_2d, save_path, n_grid=200):
    """KDE contour + density cross-section. Real clusters = valleys between peaks."""
    kde = gaussian_kde(X_2d.T, bw_method=0.08)
    xmin, xmax = X_2d[:, 0].min(), X_2d[:, 0].max()
    ymin, ymax = X_2d[:, 1].min(), X_2d[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, n_grid),
                          np.linspace(ymin, ymax, n_grid))
    pts = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(pts).reshape(n_grid, n_grid)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Full 2D KDE
    ax = axes[0]
    ax.contourf(xx, yy, zz, levels=20, cmap='viridis')
    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=0.5, alpha=0.08, color='white')
    ax.set_title("KDE density")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # Horizontal cross-section at median PC2
    ax = axes[1]
    y_mid = np.median(X_2d[:, 1])
    y_idx = np.argmin(abs(np.linspace(ymin, ymax, n_grid) - y_mid))
    ax.plot(np.linspace(xmin, xmax, n_grid), zz[y_idx, :])
    ax.set_title(f"Horizontal slice @ PC2≈{y_mid:.2f}")
    ax.set_xlabel("PC1"); ax.set_ylabel("density")
    ax.grid(alpha=0.3)

    # Max density vs minimum (valley depth)
    # Project density to 1D marginal
    ax = axes[2]
    marginal_x = zz.sum(axis=0); marginal_x /= marginal_x.sum()
    x_vals = np.linspace(xmin, xmax, n_grid)
    ax.plot(x_vals, marginal_x, color='tab:blue')
    ax.set_title("Marginal density (PC1)")
    ax.set_xlabel("PC1"); ax.set_ylabel("marginal density")
    ax.grid(alpha=0.3)

    # Find peaks and valleys
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(marginal_x, prominence=marginal_x.max() * 0.05)
    valleys, _ = find_peaks(-marginal_x, prominence=marginal_x.max() * 0.02)
    for p in peaks:
        ax.axvline(x_vals[p], color='red', linestyle='--', alpha=0.5)
    for v in valleys:
        ax.axvline(x_vals[v], color='orange', linestyle=':', alpha=0.5)
    ax.annotate(f"{len(peaks)} peaks\n{len(valleys)} valleys",
                xy=(0.02, 0.95), xycoords='axes fraction',
                verticalalignment='top', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    return {"n_peaks": len(peaks), "n_valleys": len(valleys)}


# ==================== Test 5: Baseline — gaussian null ====================

def gaussian_null_baseline(X, label, n_sub=2000):
    """Same shape sampled from gaussian — does H0 show 'fake' gap?"""
    print(f"\n--- Gaussian null for {label} ---")
    rng = np.random.default_rng(42)
    if len(X) > n_sub:
        X = X[rng.choice(len(X), n_sub, replace=False)]
    mean = X.mean(axis=0)
    cov = np.cov(X.T)
    X_null = rng.multivariate_normal(mean, cov, size=len(X))

    res = ripser(X_null, maxdim=0)
    h0 = res["dgms"][0]
    pers = h0[:, 1] - h0[:, 0]
    top = np.sort(pers[np.isfinite(pers)])[::-1][:5]
    print(f"  Gaussian H0 top5: {top.round(3)}")
    print(f"  top1/top2 ratio: {top[0]/top[1]:.3f}")
    return top


# ==================== Test 6: Downsample 20000→5000 ====================

def subsample_consistency(wp_flat, n_trials=5, n_sub=5000):
    """Does 5000-subsample of 20000 give same 3-mode structure?"""
    print(f"\n--- Subsample consistency (20000 → {n_sub} × {n_trials} trials) ---")
    rng = np.random.default_rng(42)
    all_h0_gaps = []
    all_silhouettes = []
    for trial in range(n_trials):
        idx = rng.choice(len(wp_flat), n_sub, replace=False)
        X_sub = wp_flat[idx]
        res = ripser(X_sub[rng.choice(n_sub, 1500, replace=False)], maxdim=0)
        h0 = res["dgms"][0]
        pers = h0[:, 1] - h0[:, 0]
        top = np.sort(pers[np.isfinite(pers)])[::-1][:5]
        gap_23 = top[1] / top[2]
        # Silhouette at k=3
        km = KMeans(n_clusters=3, n_init=3, random_state=trial).fit(X_sub)
        s = silhouette_score(X_sub, km.labels_, sample_size=2000,
                              random_state=trial)
        all_h0_gaps.append(gap_23)
        all_silhouettes.append(s)
        print(f"  trial {trial}: H0 top3: {top[:3].round(2)}  "
              f"gap(top2/top3)={gap_23:.2f}  silhouette(k=3)={s:.3f}")
    print(f"  gap_23: mean={np.mean(all_h0_gaps):.2f} ± {np.std(all_h0_gaps):.2f}")
    print(f"  silhouette: mean={np.mean(all_silhouettes):.3f} ± "
          f"{np.std(all_silhouettes):.3f}")
    return all_h0_gaps, all_silhouettes


# ==================== Main ====================

def main():
    print("=== Cluster Validation on 5000 vs 20000 ===\n", flush=True)

    d_5k = np.load(os.path.join(ROOT, "outputs/results/collision_free_dataset/trajectories_collision_free_5000x100x6.npy"))
    d_20k = np.load(os.path.join(ROOT, "outputs/results/collision_free_dataset/trajectories_collision_free_20000x100x6.npy"))

    wp_5k = extract_waypoints(d_5k)
    wp_20k = extract_waypoints(d_20k)

    pca_5k = PCA(n_components=2).fit_transform(wp_5k)
    pca_20k = PCA(n_components=2).fit_transform(wp_20k)

    # Test 1: Silhouette sweep ---------
    print("\n=== Test 1: Silhouette sweep ===")
    print("\nCol-Free 5000 (18D):")
    sil_5k = silhouette_sweep(wp_5k, n_sub=3000)
    print("\nCol-Free 20000 (18D):")
    sil_20k = silhouette_sweep(wp_20k, n_sub=3000)

    # Test 2: Gap statistic ---------
    print("\n=== Test 2: Gap statistic (vs uniform null) ===")
    print("\nCol-Free 5000:")
    k_r, gaps_5k, _ = gap_statistic(wp_5k, n_refs=5, n_sub=2000)
    print("\nCol-Free 20000:")
    _, gaps_20k, _ = gap_statistic(wp_20k, n_refs=5, n_sub=2000)

    # Test 3: DBSCAN ---------
    print("\n=== Test 3: DBSCAN (no k required) ===")
    print("\nCol-Free 5000:")
    db_5k = dbscan_scan(wp_5k, min_samples=20)
    print("\nCol-Free 20000:")
    db_20k = dbscan_scan(wp_20k, min_samples=30)

    # Test 4: KDE valleys ---------
    print("\n=== Test 4: KDE valley detection (2D PCA) ===")
    print("\nCol-Free 5000:")
    kde_5k = kde_valley_check(pca_5k, os.path.join(SAVE, "kde_5k.png"))
    print("\nCol-Free 20000:")
    kde_20k = kde_valley_check(pca_20k, os.path.join(SAVE, "kde_20k.png"))

    # Test 5: Gaussian null ---------
    print("\n=== Test 5: Gaussian null baseline ===")
    null_5k = gaussian_null_baseline(wp_5k, "5000")
    null_20k = gaussian_null_baseline(wp_20k, "20000")

    # Test 6: Subsample consistency ---------
    print("\n=== Test 6: Subsample 20000 → 5000 consistency ===")
    gaps_sub, sil_sub = subsample_consistency(wp_20k, n_trials=5, n_sub=5000)

    # ---- Summary plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(list(sil_5k.keys()), list(sil_5k.values()), 'o-',
            label='Col-Free 5000', color='tab:blue')
    ax.plot(list(sil_20k.keys()), list(sil_20k.values()), 's-',
            label='Col-Free 20000', color='tab:orange')
    ax.set_xlabel("k (# clusters)"); ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette sweep"); ax.legend(); ax.grid(alpha=0.3)
    ax.axvline(3, color='red', linestyle='--', alpha=0.5, label='k=3')

    ax = axes[1]
    ax.plot(k_r, gaps_5k, 'o-', label='Col-Free 5000', color='tab:blue')
    ax.plot(k_r, gaps_20k, 's-', label='Col-Free 20000', color='tab:orange')
    ax.set_xlabel("k"); ax.set_ylabel("Gap statistic")
    ax.set_title("Gap statistic (higher = better separation)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    labels = ['5k data', '5k gauss null', '20k data', '20k gauss null']
    # We compare top1/top2 ratio for data vs gaussian-null
    # Use already-computed ripser output
    from sklearn.neighbors import NearestNeighbors  # noqa
    # Recompute H0 top ratio for data
    rng = np.random.default_rng(42)
    def h0_ratio(X, n=2000):
        sub = rng.choice(len(X), min(n, len(X)), replace=False)
        r = ripser(X[sub], maxdim=0)
        h0 = r["dgms"][0]; p = h0[:, 1] - h0[:, 0]
        top = np.sort(p[np.isfinite(p)])[::-1][:3]
        return top[1] / top[2]   # gap at top2→top3
    ratios = [h0_ratio(wp_5k), null_5k[1]/null_5k[2],
              h0_ratio(wp_20k), null_20k[1]/null_20k[2]]
    colors = ['tab:blue', 'gray', 'tab:orange', 'gray']
    ax.bar(labels, ratios, color=colors)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel("H0 gap ratio (top2/top3)")
    ax.set_title("Real gap vs gaussian-null")
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.02, f"{r:.2f}", ha='center', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE, "validation_summary.png"), dpi=150)
    plt.close()
    print(f"\nSaved: {os.path.join(SAVE, 'validation_summary.png')}")

    # Print final verdict
    print(f"\n{'='*70}")
    print("VERDICT TABLE")
    print(f"{'='*70}")
    print(f"{'Test':<40} {'5000':>12} {'20000':>12}")
    print("-" * 70)
    print(f"{'Silhouette k=2':<40} {sil_5k[2]:>12.3f} {sil_20k[2]:>12.3f}")
    print(f"{'Silhouette k=3':<40} {sil_5k[3]:>12.3f} {sil_20k[3]:>12.3f}")
    print(f"{'Silhouette k=4':<40} {sil_5k[4]:>12.3f} {sil_20k[4]:>12.3f}")
    print(f"{'Silhouette best k':<40} "
          f"{max(sil_5k, key=sil_5k.get):>12d} "
          f"{max(sil_20k, key=sil_20k.get):>12d}")
    print(f"{'Gap stat max-k':<40} "
          f"{k_r[int(np.argmax(gaps_5k))]:>12d} "
          f"{k_r[int(np.argmax(gaps_20k))]:>12d}")
    print(f"{'DBSCAN (median eps) # clusters':<40} "
          f"{db_5k[0]['n_clusters']:>12d} "
          f"{db_20k[0]['n_clusters']:>12d}")
    print(f"{'KDE marginal peaks (PC1)':<40} "
          f"{kde_5k['n_peaks']:>12d} {kde_20k['n_peaks']:>12d}")
    print(f"{'H0 gap top2/top3 (data)':<40} "
          f"{ratios[0]:>12.2f} {ratios[2]:>12.2f}")
    print(f"{'H0 gap top2/top3 (gauss-null)':<40} "
          f"{ratios[1]:>12.2f} {ratios[3]:>12.2f}")
    print(f"{'Subsample 20k→5k gap mean':<40} "
          f"{'':>12} {np.mean(gaps_sub):>12.2f}")
    print(f"{'Subsample 20k→5k silhouette mean':<40} "
          f"{'':>12} {np.mean(sil_sub):>12.3f}")

    np.save(os.path.join(SAVE, "validation_results.npy"), {
        "silhouette_5k": sil_5k, "silhouette_20k": sil_20k,
        "gaps_5k": gaps_5k, "gaps_20k": gaps_20k,
        "db_5k": db_5k, "db_20k": db_20k,
        "kde_5k": kde_5k, "kde_20k": kde_20k,
        "h0_ratios": ratios, "subsample_gaps": gaps_sub,
        "subsample_silhouettes": sil_sub,
    })
    print(f"\n=== Done. Output: {SAVE} ===")


if __name__ == "__main__":
    main()
