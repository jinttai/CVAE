"""
Reachable Orientation Set 계산.

S2 sphere를 ~500 방향으로 등분하고,
각 방향(axis)에 대해 도달 가능한 최대 회전 각도를 LBFGS로 구한다.

Loss 설계:
  1) -angle_projected  : 원하는 축 방향으로 투영된 회전각을 최대화
  2) axis_deviation    : 실제 회전축이 원하는 방향에서 벗어나는 정도 벌점
  3) joint_limit_barrier: 관절 한계 근처에서 부드러운 벌점 (log barrier)
  4) joint_smoothness   : 웨이포인트 변화량 벌점

Joint limits (SC_ur10e):
  shoulder_pan  : [-6.28, 6.28]  (실질적으로 무제한)
  shoulder_lift : [-6.28, 6.28]
  elbow         : [-3.14, 3.14]
  wrist_1       : [-6.28, 6.28]
  wrist_2       : [-6.28, 6.28]
  wrist_3       : [-6.28, 6.28]

  다만 CVAE 학습에서 [-2.44, 2.44]로 제한하므로 여기서도 동일하게 적용.

Usage:
    python -m src.reachability.compute_reachable_set
    python -m src.reachability.compute_reachable_set --n_dirs 500 --n_restarts 8
"""

import os
import sys
import math
import argparse
import time

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from src.utils.runtime_env import configure_windows_runtime
configure_windows_runtime()

import torch
import torch.optim as optim
import numpy as np
from torch.func import vmap

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


# =====================================================================
# Joint Limits
# =====================================================================
# URDF 기준 한계 (SC_ur10e.urdf)
JOINT_LIMITS_URDF = torch.tensor([
    [-6.283185, 6.283185],   # shoulder_pan
    [-6.283185, 6.283185],   # shoulder_lift
    [-3.141590, 3.141590],   # elbow
    [-6.283185, 6.283185],   # wrist_1
    [-6.283185, 6.283185],   # wrist_2
    [-6.283185, 6.283185],   # wrist_3
])

# CVAE 학습과 동일한 실용적 한계
JOINT_LIMITS = torch.tensor([
    [-2.44, 2.44],
    [-2.44, 2.44],
    [-2.44, 2.44],
    [-2.44, 2.44],
    [-2.44, 2.44],
    [-2.44, 2.44],
])


# =====================================================================
# S2 Sphere 등분 (Fibonacci Lattice)
# =====================================================================
def fibonacci_sphere(n: int) -> np.ndarray:
    """Fibonacci lattice로 S2 위 n개 점을 균등 분포.

    Returns:
        directions: [n, 3] unit vectors
    """
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    indices = np.arange(n, dtype=np.float64)

    theta = 2.0 * np.pi * indices / golden_ratio  # azimuthal
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)  # polar

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack([x, y, z], axis=-1).astype(np.float32)


# =====================================================================
# Loss 함수
# =====================================================================
class ReachabilityLoss:
    """단일 방향에 대한 최대 도달 각도 탐색 Loss.

    Args:
        physics: PhysicsLayer 인스턴스
        direction: [3] 목표 회전축 (unit vector)
        joint_limits: [n_q, 2] (min, max)
        weights: dict of loss weights
    """

    def __init__(self, physics: PhysicsLayer, direction: torch.Tensor,
                 joint_limits: torch.Tensor, weights: dict):
        self.physics = physics
        self.direction = direction  # [3] unit
        self.joint_limits = joint_limits.to(physics.device)  # [n_q, 2]
        self.w = weights
        self.device = physics.device
        self.n_q = physics.n_q
        self.num_waypoints = physics.num_waypoints

        # 초기 자세: identity
        self.q0_init = torch.tensor([0., 0., 0., 1.], device=self.device)

    def __call__(self, waypoints_flat: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            waypoints_flat: [num_waypoints * n_q]
        Returns:
            loss: scalar
            info: dict with components
        """
        batch_wp = waypoints_flat.unsqueeze(0)  # [1, dim]
        q_traj, q_dot_traj = self.physics.generate_trajectory(batch_wp)
        q_t = q_traj[0]      # [T, n_q]
        qd_t = q_dot_traj[0]  # [T, n_q]

        # ── 시뮬레이션: R_final 구하기 ──
        R_init = torch.eye(3, device=self.device)
        dt = self.physics.dt

        all_wb = vmap(self.physics._compute_wb)(q_t, qd_t)
        all_R_delta = vmap(lambda w: self.physics._rot_from_omega(w, dt))(all_wb)

        R_curr = R_init
        for t in range(self.physics.num_steps):
            R_curr = R_curr @ all_R_delta[t]

        # ── SO(3) log map → rotation vector ──
        rot_vec = self._so3_log(R_curr)  # [3]
        angle = torch.linalg.norm(rot_vec)

        # ── Loss 1: 투영 각도 최대화 (negative = minimize) ──
        # dot(rot_vec, direction) = angle * cos(axis와 direction 사이각)
        # 이걸 최대화하면 자연스럽게 축 정렬 + 각도 최대화 동시 달성
        angle_projected = torch.dot(rot_vec, self.direction)
        loss_angle = -angle_projected

        # ── Loss 2: 축 편차 벌점 ──
        # angle이 작을 때 axis가 ill-defined → gradient 신호 없음
        # 해결: rot_vec 자체를 direction 방향과 비교 (angle=0이면 자연스럽게 0)
        # cross_component = ||rot_vec - (rot_vec·d)d|| = 원하는 축에서 벗어난 회전 성분
        proj_on_d = angle_projected * self.direction  # direction 방향 성분
        cross_component = rot_vec - proj_on_d         # 직교 성분
        loss_axis = torch.dot(cross_component, cross_component)  # 직교 성분의 크기²
        # angle=0이면 loss_axis=0 → gradient가 angle 쪽으로만 흐름 (stuck 방지)

        # ── Loss 3: Joint limit barrier (soft) ──
        waypoints = waypoints_flat.view(self.num_waypoints, self.n_q)
        q_min = self.joint_limits[:, 0]
        q_max = self.joint_limits[:, 1]
        margin = 0.05  # rad, barrier 시작 거리

        # 상한/하한까지 거리 (양수 = 안전)
        dist_lo = waypoints - q_min.unsqueeze(0) - margin
        dist_hi = q_max.unsqueeze(0) - waypoints - margin

        # log barrier: -log(dist) when dist > 0, else large penalty
        barrier_lo = torch.where(dist_lo > 0, -torch.log(dist_lo + 1e-8), 100.0 * (-dist_lo + 1.0))
        barrier_hi = torch.where(dist_hi > 0, -torch.log(dist_hi + 1e-8), 100.0 * (-dist_hi + 1.0))
        loss_barrier = (barrier_lo.mean() + barrier_hi.mean())


        # ── Total ──
        total = (self.w["angle"] * loss_angle
                 + self.w["axis"] * loss_axis
                 + self.w["barrier"] * loss_barrier
                 )

        info = {
            "angle_projected": angle_projected.item(),
            "angle_total": angle.item(),
            "cross_component": loss_axis.item(),
            "loss_angle": loss_angle.item(),
            "loss_axis": loss_axis.item(),
            "loss_barrier": loss_barrier.item(),
            "total": total.item(),
        }
        return total, info

    @staticmethod
    def _so3_log(R: torch.Tensor) -> torch.Tensor:
        """SO(3) log map: R → rotation vector [3]."""
        trace_val = R[0, 0] + R[1, 1] + R[2, 2]
        cos_theta = torch.clamp((trace_val - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)

        factor = torch.where(
            theta.abs() < 1e-6,
            torch.ones_like(theta) * 0.5,
            theta / (2.0 * sin_theta + 1e-12),
        )
        omega = factor * torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        return omega


# =====================================================================
# 단일 방향 최적화
# =====================================================================
def optimize_direction(physics, direction, joint_limits, weights,
                       n_restarts=8, lbfgs_iter=80):
    """하나의 S2 방향에 대해 최대 도달 각도 탐색.

    multi-restart LBFGS: 여러 초기 추측에서 시작하여 최선 선택.

    Returns:
        best_angle: float (rad)
        best_waypoints: [num_waypoints * n_q]
    """
    device = physics.device
    n_q = physics.n_q
    num_wp = physics.num_waypoints
    dim = num_wp * n_q

    d = torch.tensor(direction, device=device, dtype=torch.float32)
    loss_fn = ReachabilityLoss(physics, d, joint_limits, weights)

    best_angle = -float("inf")
    best_wp = None

    # tanh reparameterization: wp_actual = mid + scale * tanh(raw)
    # → wp_actual ∈ (q_min, q_max) 구조적 보장
    q_min = joint_limits[:, 0].to(device)
    q_max = joint_limits[:, 1].to(device)
    mid = (q_min + q_max) / 2.0       # [n_q]
    scale = (q_max - q_min) / 2.0     # [n_q]
    mid_rep = mid.repeat(num_wp)      # [dim]
    scale_rep = scale.repeat(num_wp)  # [dim]

    def raw_to_waypoints(raw):
        return mid_rep + scale_rep * torch.tanh(raw)

    for restart in range(n_restarts):
        if restart == 0:
            raw = torch.empty(dim, device=device).uniform_(-0.3, 0.3)
        else:
            raw = torch.empty(dim, device=device).uniform_(-1.5, 1.5)

        raw = raw.requires_grad_(True)
        optimizer = optim.LBFGS(
            [raw], lr=1.0, max_iter=lbfgs_iter,
            history_size=50, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            wp_actual = raw_to_waypoints(raw)
            loss, _ = loss_fn(wp_actual)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            wp_actual = raw_to_waypoints(raw)
            _, info = loss_fn(wp_actual)

        if info["angle_projected"] > best_angle:
            best_angle = info["angle_projected"]
            best_wp = wp_actual.detach().clone()

    return best_angle, best_wp


# =====================================================================
# 전체 Reachable Set 계산
# =====================================================================
def compute_reachable_set(
    n_dirs=500,
    n_restarts=8,
    lbfgs_iter=80,
    device="cuda",
    save_path=None,
):
    """S2 sphere를 n_dirs 방향으로 등분, 각 방향의 최대 도달 각도를 계산.

    Returns:
        directions: [n_dirs, 3]
        max_angles: [n_dirs] (rad)
    """
    # ── Robot & Physics ──
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)

    num_waypoints = 3
    total_time = 10.0
    physics = PhysicsLayer(robot, num_waypoints, total_time, device)

    joint_limits = JOINT_LIMITS.to(device)

    # ── Loss weights ──
    weights = {
        "angle": 1.0,       # 각도 최대화 (음수 loss → 최대화)
        "axis": 5.0,        # 축 정렬 벌점
        "barrier": 0.001,   # joint limit barrier
        "smooth": 0.01,     # 웨이포인트 스무딩
    }

    # ── S2 방향들 ──
    directions = fibonacci_sphere(n_dirs)

    max_angles = np.zeros(n_dirs, dtype=np.float32)
    all_waypoints = []

    print(f"Computing reachable set: {n_dirs} directions, "
          f"{n_restarts} restarts, device={device}")
    print(f"Joint limits: [{JOINT_LIMITS[0, 0]:.2f}, {JOINT_LIMITS[0, 1]:.2f}] rad")
    print(f"Loss weights: {weights}")
    print("=" * 70)

    t_start = time.time()

    for i in range(n_dirs):
        angle, wp = optimize_direction(
            physics, directions[i], joint_limits, weights,
            n_restarts=n_restarts, lbfgs_iter=lbfgs_iter,
        )
        max_angles[i] = angle
        all_waypoints.append(wp.cpu().numpy())

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_dirs - i - 1)
            print(f"[{i+1:4d}/{n_dirs}]  angle={np.degrees(angle):6.2f}°  "
                  f"mean={np.degrees(max_angles[:i+1].mean()):6.2f}°  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    elapsed_total = time.time() - t_start
    print("=" * 70)
    print(f"Done in {elapsed_total:.1f}s")
    print(f"Angle stats: mean={np.degrees(max_angles.mean()):.2f}°  "
          f"min={np.degrees(max_angles.min()):.2f}°  "
          f"max={np.degrees(max_angles.max()):.2f}°")

    # ── Save ──
    if save_path is None:
        save_path = os.path.join(ROOT_DIR, "outputs/results/reachable_set")
    os.makedirs(save_path, exist_ok=True)

    np.savez(
        os.path.join(save_path, "reachable_set.npz"),
        directions=directions,
        max_angles=max_angles,
        waypoints=np.stack(all_waypoints),
        joint_limits=JOINT_LIMITS.numpy(),
        weights=weights,
        n_restarts=n_restarts,
        lbfgs_iter=lbfgs_iter,
    )
    print(f"Saved to {save_path}/reachable_set.npz")

    # ── Visualization ──
    plot_path = os.path.join(save_path, "reachable_set.png")
    plot_reachable_set(directions, max_angles, plot_path)

    return directions, max_angles


# =====================================================================
# Visualization
# =====================================================================
def plot_reachable_set(directions, max_angles, save_path=None, npz_path=None):
    """S2 sphere 위에 각 방향별 최대 도달 각도를 직선(ray)으로 시각화.

    각 방향 d에 대해 원점에서 d * (normalized_angle) 길이의 직선을 그린다.
    색상은 각도 크기를 나타낸다.

    Args:
        directions: [N, 3] unit vectors
        max_angles: [N] rad
        save_path: 저장 경로 (None이면 표시만)
        npz_path: .npz 파일에서 로드 (directions, max_angles 대신)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import Normalize
    from matplotlib import cm

    if npz_path is not None:
        data = np.load(npz_path, allow_pickle=True)
        directions = data["directions"]
        max_angles = data["max_angles"]

    angles_deg = np.degrees(max_angles)

    # 정규화: 0 ~ max_angle → 0 ~ 1 (ray 길이)
    angle_max = angles_deg.max() if angles_deg.max() > 0 else 1.0
    lengths = angles_deg / angle_max  # [0, 1]

    # 음수 각도(반대 방향) 처리: 최소 0
    lengths = np.clip(lengths, 0, None)

    # ray 끝점: origin → direction * length
    endpoints = directions * lengths[:, None]

    # colormap
    norm = Normalize(vmin=0, vmax=angle_max)
    cmap = cm.hot_r
    colors = cmap(norm(angles_deg))

    # ── 3D Plot ──
    fig = plt.figure(figsize=(16, 6))

    # --- View 1: 3D perspective ---
    ax1 = fig.add_subplot(131, projection="3d")
    _draw_sphere_rays(ax1, directions, endpoints, colors, norm, cmap, angle_max)
    ax1.set_title("3D View")
    ax1.view_init(elev=25, azim=45)

    # --- View 2: top (Z axis) ---
    ax2 = fig.add_subplot(132, projection="3d")
    _draw_sphere_rays(ax2, directions, endpoints, colors, norm, cmap, angle_max)
    ax2.set_title("Top View (Z-axis)")
    ax2.view_init(elev=90, azim=0)

    # --- View 3: front (Y axis) ---
    ax3 = fig.add_subplot(133, projection="3d")
    _draw_sphere_rays(ax3, directions, endpoints, colors, norm, cmap, angle_max)
    ax3.set_title("Front View (Y-axis)")
    ax3.view_init(elev=0, azim=0)

    # colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3], shrink=0.6, pad=0.08)
    cbar.set_label("Max reachable angle (deg)")

    fig.suptitle(
        f"Orientation Reachable Set  |  {len(directions)} dirs  |  "
        f"mean={angles_deg.mean():.1f}°  min={angles_deg.min():.1f}°  max={angles_deg.max():.1f}°",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.02, right=0.88, wspace=0.05)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.close(fig)


def _draw_sphere_rays(ax, directions, endpoints, colors, norm, cmap, angle_max):
    """하나의 3D axes에 reference sphere + rays를 그린다."""
    # reference unit sphere (wireframe)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgray", alpha=0.15, linewidth=0.3)

    # rays: 원점 → endpoint
    for i in range(len(directions)):
        ax.plot(
            [0, endpoints[i, 0]],
            [0, endpoints[i, 1]],
            [0, endpoints[i, 2]],
            color=colors[i],
            linewidth=0.8,
            alpha=0.7,
        )

    # axis labels
    lim = 1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")


# =====================================================================
# main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Compute orientation reachable set on S2")
    parser.add_argument("--n_dirs", type=int, default=500)
    parser.add_argument("--n_restarts", type=int, default=8)
    parser.add_argument("--lbfgs_iter", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    compute_reachable_set(
        n_dirs=args.n_dirs,
        n_restarts=args.n_restarts,
        lbfgs_iter=args.lbfgs_iter,
        device=args.device,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
