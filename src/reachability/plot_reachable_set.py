"""
Reachable Set 시각화 (독립 실행).

Usage:
    python -m src.reachability.plot_reachable_set
    python -m src.reachability.plot_reachable_set --npz outputs/results/reachable_set/reachable_set.npz
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize
from matplotlib import cm


def plot_reachable_set(directions, max_angles, save_path, title_extra=""):
    """S2 위 각 방향별 최대 도달 각도를 ray로 시각화.

    원점에서 direction * (angle / max_angle) 길이의 직선을 그린다.

    Args:
        directions: [N, 3] unit vectors
        max_angles: [N] rad
        save_path: 저장 경로
        title_extra: 제목에 추가할 문자열
    """
    angles_deg = np.degrees(max_angles)
    angle_max = angles_deg.max() if angles_deg.max() > 0 else 1.0
    lengths = np.clip(angles_deg / angle_max, 0, None)
    endpoints = directions * lengths[:, None]

    norm = Normalize(vmin=0, vmax=angle_max)
    cmap = cm.hot_r
    colors = cmap(norm(angles_deg))

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(131, projection="3d")
    _draw_sphere_rays(ax1, endpoints, colors)
    ax1.set_title("3D View")
    ax1.view_init(elev=25, azim=45)

    ax2 = fig.add_subplot(132, projection="3d")
    _draw_sphere_rays(ax2, endpoints, colors)
    ax2.set_title("Top View (Z-axis)")
    ax2.view_init(elev=90, azim=0)

    ax3 = fig.add_subplot(133, projection="3d")
    _draw_sphere_rays(ax3, endpoints, colors)
    ax3.set_title("Front View (Y-axis)")
    ax3.view_init(elev=0, azim=0)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3], shrink=0.6, pad=0.08)
    cbar.set_label("Max reachable angle (deg)")

    fig.suptitle(
        f"Orientation Reachable Set  |  {len(directions)} dirs  |  "
        f"mean={angles_deg.mean():.1f}\u00b0  min={angles_deg.min():.1f}\u00b0  "
        f"max={angles_deg.max():.1f}\u00b0  {title_extra}",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.02, right=0.88, wspace=0.05)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def _draw_sphere_rays(ax, endpoints, colors):
    """Unit sphere wireframe + rays."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgray", alpha=0.15, linewidth=0.3)

    for i in range(len(endpoints)):
        ax.plot(
            [0, endpoints[i, 0]],
            [0, endpoints[i, 1]],
            [0, endpoints[i, 2]],
            color=colors[i], linewidth=0.8, alpha=0.7,
        )

    lim = 1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")


def main():
    ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
    default_npz = os.path.join(ROOT_DIR, "outputs/results/reachable_set/reachable_set.npz")

    parser = argparse.ArgumentParser(description="Plot orientation reachable set from .npz")
    parser.add_argument("--npz", type=str, default=default_npz)
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: same dir as npz)")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    directions = data["directions"]
    max_angles = data["max_angles"]

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.npz), "reachable_set.png")

    plot_reachable_set(directions, max_angles, args.output)


if __name__ == "__main__":
    main()
