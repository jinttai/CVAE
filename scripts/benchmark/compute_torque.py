import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

# Add root directory to sys.path to find src
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

import src.dynamics.spart_functions_torch as spart  # noqa: E402
from src.dynamics.urdf2robot_torch import urdf2robot  # noqa: E402
from src.models.cvae import CVAE  # noqa: E402
from src.training.physics_layer import PhysicsLayer  # noqa: E402


def euler_to_quaternion(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)
    Using ZYX convention (yaw around Z, pitch around Y, roll around X)
    """
    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return torch.stack([qx, qy, qz, qw], dim=-1)


def generate_random_quaternion_from_euler(
    batch_size: int, max_angle_deg: float, device: str, dtype: torch.dtype
) -> torch.Tensor:
    max_angle_rad = math.radians(max_angle_deg)
    roll = (2 * max_angle_rad) * torch.rand(batch_size, device=device, dtype=dtype) - max_angle_rad
    pitch = (2 * max_angle_rad) * torch.rand(batch_size, device=device, dtype=dtype) - max_angle_rad
    yaw = (2 * max_angle_rad) * torch.rand(batch_size, device=device, dtype=dtype) - max_angle_rad
    return euler_to_quaternion(roll, pitch, yaw)


def save_csv(path: str, header: List[str], rows: List[List[float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # pragma: no cover
        return None


def finite_diff_second(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Second derivative along time axis (dim=0) using finite differences.
    x: [T, D]
    returns x_ddot: [T, D]
    """
    T = x.size(0)
    if T < 3:
        return torch.zeros_like(x)

    x_dd = torch.zeros_like(x)
    # central difference for interior
    x_dd[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / (dt * dt)
    # forward/backward for boundaries (2nd order)
    x_dd[0] = (x[2] - 2 * x[1] + x[0]) / (dt * dt)
    x_dd[-1] = (x[-1] - 2 * x[-2] + x[-3]) / (dt * dt)
    return x_dd


def generate_trajectory_with_accel(
    physics: PhysicsLayer, waypoints_flat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PhysicsLayer.generate_trajectory()와 동일한 4분절 5차 다항식(quintic) 궤적을 사용하되,
    basis를 해석적으로 미분해서 q_ddot까지 반환.

    Quintic basis (t in [0,1]):
      - position basis:      b(t)  = 6 t^5 - 15 t^4 + 10 t^3
      - velocity basis:      b'(t) = 30 t^4 - 60 t^3 + 30 t^2
      - accel basis:         b''(t)= 120 t^3 - 180 t^2 + 60 t

    여기서 t는 분절 내 정규화 시간 [0,1].
    실제 시간 미분은 segment_time 스케일을 적용:
      q_dot = (dq/dt_norm) / segment_time
      q_ddot = (d2q/dt_norm2) / segment_time^2
    """
    device = physics.device
    dtype = waypoints_flat.dtype

    batch_size = waypoints_flat.size(0)
    n_q = physics.n_q
    num_waypoints = physics.num_waypoints

    w_mid = waypoints_flat.view(batch_size, num_waypoints, n_q)
    zeros = torch.zeros(batch_size, 1, n_q, device=device, dtype=dtype)
    w_full = torch.cat([zeros, w_mid, zeros], dim=1)  # [B, 5, n_q]

    num_segments = num_waypoints + 1  # 4
    steps_per_segment = physics.num_steps // num_segments
    remainder = physics.num_steps % num_segments

    q_traj = torch.zeros(batch_size, physics.num_steps, n_q, device=device, dtype=dtype)
    q_dot_traj = torch.zeros(batch_size, physics.num_steps, n_q, device=device, dtype=dtype)
    q_ddot_traj = torch.zeros(batch_size, physics.num_steps, n_q, device=device, dtype=dtype)

    step_idx = 0
    segment_time = physics.total_time / num_segments
    inv_seg_t = 1.0 / segment_time
    inv_seg_t2 = inv_seg_t * inv_seg_t

    for seg in range(num_segments):
        q_start = w_full[:, seg, :]  # [B, n_q]
        q_end = w_full[:, seg + 1, :]  # [B, n_q]
        dq = (q_end - q_start).unsqueeze(1)  # [B, 1, n_q]

        seg_steps = steps_per_segment + (1 if seg < remainder else 0)
        t_seg = torch.linspace(0, 1, seg_steps, device=device, dtype=dtype)  # [seg_steps]

        # quintic position basis
        t = t_seg
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        basis = 6.0 * t5 - 15.0 * t4 + 10.0 * t3  # [seg_steps]
        q_seg = q_start.unsqueeze(1) + dq * basis.unsqueeze(0).unsqueeze(-1)

        # velocity (w.r.t normalized time)
        d_basis = 30.0 * t4 - 60.0 * t3 + 30.0 * t2  # [seg_steps]
        qd_seg = dq * d_basis.unsqueeze(0).unsqueeze(-1)
        qd_seg = qd_seg * inv_seg_t

        # acceleration (w.r.t normalized time)
        dd_basis = 120.0 * t3 - 180.0 * t2 + 60.0 * t  # [seg_steps]
        qdd_seg = dq * dd_basis.unsqueeze(0).unsqueeze(-1)
        qdd_seg = qdd_seg * inv_seg_t2

        q_traj[:, step_idx : step_idx + seg_steps, :] = q_seg
        q_dot_traj[:, step_idx : step_idx + seg_steps, :] = qd_seg
        q_ddot_traj[:, step_idx : step_idx + seg_steps, :] = qdd_seg
        step_idx += seg_steps

    return q_traj, q_dot_traj, q_ddot_traj


def build_cvae(
    *,
    weights: str,
    cond_dim: int,
    output_dim: int,
    latent_dim: int,
    joint_limits: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    untrained: bool,
    untrained_seed: int,
    save_untrained_pth: str,
    out_dir: str,
) -> CVAE:
    if untrained:
        torch.manual_seed(int(untrained_seed))
        model = CVAE(cond_dim, output_dim, latent_dim, joint_limits=joint_limits).to(device=device, dtype=dtype)
        model.eval()
        save_path = save_untrained_pth.strip()
        if not save_path:
            save_path = os.path.join(out_dir, f"untrained_seed{int(untrained_seed)}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"[untrained] saved random-init weights to: {save_path}")
        return model

    if not os.path.exists(weights):
        raise FileNotFoundError(f"weights not found: {weights}")

    model = CVAE(cond_dim, output_dim, latent_dim, joint_limits=joint_limits).to(device=device, dtype=dtype)
    state = torch.load(weights, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"[Warning] strict=True failed, retry strict=False\n  {e}")
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def compute_tau_trajectory(
    *,
    robot: Dict,
    q_traj: torch.Tensor,  # [T, n_q]
    qd_traj: torch.Tensor,  # [T, n_q]
    qdd_traj: torch.Tensor,  # [T, n_q]
    damping: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute joint torque trajectory tau(t) using SPART H/C matrices (free-floating EoM, no external wrenches).

    For each step:
      - compute H0, H0m, Hm from configuration
      - compute base twist u0 from non-holonomic constraint: (H0 + dI) u0 = -H0m * um
      - compute convective matrices C0, C0m, Cm0, Cm from twists
      - compute base acceleration u0dot from base EoM with zero base wrench:
            (H0 + dI) u0dot = -(H0m umdot + C0 u0 + C0m um)
      - joint torque:
            tau = H0m^T u0dot + Hm umdot + Cm0 u0 + Cm um

    Returns:
      tau_traj: [T, n_q]
      u0_traj:  [T, 6]
      u0dot_traj: [T, 6]
    """
    device = q_traj.device
    dtype = q_traj.dtype

    n_q = int(robot["n_q"])
    T = int(q_traj.size(0))
    eye6 = torch.eye(6, device=device, dtype=dtype)

    R0 = torch.eye(3, device=device, dtype=dtype)
    r0 = torch.zeros(3, device=device, dtype=dtype)

    tau_list: List[torch.Tensor] = []
    u0_list: List[torch.Tensor] = []
    u0dot_list: List[torch.Tensor] = []

    for t in range(T):
        qm = q_traj[t]
        um = qd_traj[t]
        umdot = qdd_traj[t]

        # Kinematics and inertia terms (depend on qm)
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = spart.inertia_projection(R0, RL, robot)
        M0_tilde, Mm_tilde = spart.mass_composite_body(I0, Im, Bij, Bi0, robot)
        H0, H0m, Hm = spart.generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)

        # Base twist from constraint (match PhysicsLayer logic)
        H0_damped = H0 + float(damping) * eye6
        rhs_u0 = -(H0m @ um)  # [6]
        u0 = torch.linalg.solve(H0_damped, rhs_u0)  # [6]

        # Twists for convective terms
        t0, tL = spart.velocities(Bij, Bi0, P0, pm, u0, um, robot)
        C0, C0m, Cm0, Cm = spart.convective_inertia_matrix(
            t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot
        )

        # Base acceleration from base EoM with zero base wrench
        rhs_u0dot = -(H0m @ umdot + C0 @ u0 + C0m @ um)  # [6]
        u0dot = torch.linalg.solve(H0_damped, rhs_u0dot)  # [6]

        # Joint torque
        tau = (H0m.T @ u0dot) + (Hm @ umdot) + (Cm0 @ u0) + (Cm @ um)  # [n_q]

        tau_list.append(tau)
        u0_list.append(u0)
        u0dot_list.append(u0dot)

    tau_traj = torch.stack(tau_list, dim=0)
    u0_traj = torch.stack(u0_list, dim=0)
    u0dot_traj = torch.stack(u0dot_list, dim=0)
    return tau_traj, u0_traj, u0dot_traj


def plot_joint_series(t: List[float], y: torch.Tensor, title: str, ylabel: str, out_path: str) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return

    y_np = y.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    for j in range(y_np.shape[1]):
        ax.plot(t, y_np[:, j], label=f"J{j+1}")
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc="right", fontsize="small")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute qddot and joint torque trajectory, and plot torque curves")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    # Model/physics config (match training defaults)
    p.add_argument("--weights", type=str, default=os.path.join(ROOT_DIR, "outputs/weights/cvae_debug/v4.pth"))
    p.add_argument("--cond-dim", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--num-waypoints", type=int, default=3)
    p.add_argument("--total-time", type=float, default=10.0)
    p.add_argument("--max-goal-angle-deg", type=float, default=60.0)

    # Choose a sample from latent space
    p.add_argument("--num-z-samples", type=int, default=5, help="sample multiple z and pick best by physics loss")
    p.add_argument("--seed", type=int, default=0)

    # Untrained option (same architecture, random init weights)
    p.add_argument("--untrained", action="store_true")
    p.add_argument("--untrained-seed", type=int, default=0)
    p.add_argument("--save-untrained-pth", type=str, default="")

    # Output
    p.add_argument("--out-dir", type=str, default=os.path.join(ROOT_DIR, "outputs/results/torque"))

    args = p.parse_args()
    device = args.device
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    torch.manual_seed(int(args.seed))
    os.makedirs(args.out_dir, exist_ok=True)

    # Robot + physics
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    n_q = int(robot["n_q"])
    output_dim = int(args.num_waypoints) * n_q
    physics = PhysicsLayer(robot, int(args.num_waypoints), float(args.total_time), device)

    # Condition (single)
    q0_start = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)
    q0_goal = generate_random_quaternion_from_euler(1, float(args.max_goal_angle_deg), device=device, dtype=dtype)
    cond = torch.cat([q0_start, q0_goal], dim=1)

    # Model
    model = build_cvae(
        weights=args.weights,
        cond_dim=int(args.cond_dim),
        output_dim=output_dim,
        latent_dim=int(args.latent_dim),
        joint_limits=robot["joint_limits"],
        device=device,
        dtype=dtype,
        untrained=bool(args.untrained),
        untrained_seed=int(args.untrained_seed),
        save_untrained_pth=str(args.save_untrained_pth),
        out_dir=args.out_dir,
    )

    # Pick best waypoint among z samples by physics loss
    with torch.no_grad():
        K = int(args.num_z_samples)
        z = torch.randn(K, int(args.latent_dim), device=device, dtype=dtype)
        cond_batch = cond.repeat(K, 1)
        wp_batch = model.decode(cond_batch, z)  # [K, W*n_q]
        # physics loss only (same metric as optimize script)
        q_traj_b, q_dot_traj_b = physics.generate_trajectory(wp_batch)
        batch_sim_fn = torch.func.vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        loss_k, _ = batch_sim_fn(
            q_traj_b,
            q_dot_traj_b,
            q0_start.repeat(K, 1),
            q0_goal.repeat(K, 1),
        )
        loss_k = torch.where(torch.isfinite(loss_k), loss_k, torch.full_like(loss_k, float("inf")))
        best_idx = int(torch.argmin(loss_k).item())
        waypoints = wp_batch[best_idx : best_idx + 1].clone()
        best_loss = float(loss_k[best_idx].item())
        print(f"Selected best z among {K}: physics_loss={best_loss:.6f}")

        q_traj, q_dot_traj, q_ddot_traj = generate_trajectory_with_accel(physics, waypoints)
        q_traj = q_traj[0]  # [T, n_q]
        q_dot_traj = q_dot_traj[0]  # [T, n_q]
        q_ddot_traj = q_ddot_traj[0]  # [T, n_q]

    dt = float(physics.dt)

    # tau from SPART dynamics terms
    tau_traj, u0_traj, u0dot_traj = compute_tau_trajectory(
        robot=robot,
        q_traj=q_traj,
        qd_traj=q_dot_traj,
        qdd_traj=q_ddot_traj,
        damping=1e-6,
    )

    # Save CSVs
    T = int(q_traj.size(0))
    t_arr = [i * dt for i in range(T)]

    # torque csv
    torque_csv = os.path.join(args.out_dir, "torque.csv")
    header_tau = ["t"] + [f"tau_J{i+1}" for i in range(n_q)]
    rows_tau = [[float(t_arr[i])] + [float(v) for v in tau_traj[i].detach().cpu().tolist()] for i in range(T)]
    save_csv(torque_csv, header_tau, rows_tau)

    # q, qdot, qddot csv (single file)
    traj_csv = os.path.join(args.out_dir, "traj_q_qdot_qddot.csv")
    header_traj = ["t"] + [f"q_J{i+1}" for i in range(n_q)] + [f"qd_J{i+1}" for i in range(n_q)] + [
        f"qdd_J{i+1}" for i in range(n_q)
    ]
    rows_traj = []
    q_cpu = q_traj.detach().cpu()
    qd_cpu = q_dot_traj.detach().cpu()
    qdd_cpu = q_ddot_traj.detach().cpu()
    for i in range(T):
        rows_traj.append(
            [float(t_arr[i])]
            + [float(v) for v in q_cpu[i].tolist()]
            + [float(v) for v in qd_cpu[i].tolist()]
            + [float(v) for v in qdd_cpu[i].tolist()]
        )
    save_csv(traj_csv, header_traj, rows_traj)

    # base twist csv
    base_csv = os.path.join(args.out_dir, "base_u0_u0dot.csv")
    header_base = ["t"] + [f"u0_{k}" for k in range(6)] + [f"u0dot_{k}" for k in range(6)]
    u0_cpu = u0_traj.detach().cpu()
    u0dot_cpu = u0dot_traj.detach().cpu()
    rows_base = []
    for i in range(T):
        rows_base.append([float(t_arr[i])] + [float(v) for v in u0_cpu[i].tolist()] + [float(v) for v in u0dot_cpu[i].tolist()])
    save_csv(base_csv, header_base, rows_base)

    print(f"Saved: {torque_csv}")
    print(f"Saved: {traj_csv}")
    print(f"Saved: {base_csv}")

    # Plots
    plot_joint_series(t_arr, tau_traj, "Joint torque over time", "tau", os.path.join(args.out_dir, "torque.png"))
    plot_joint_series(
        t_arr, q_ddot_traj, "Joint acceleration over time (qddot)", "qddot [rad/s^2]", os.path.join(args.out_dir, "qddot.png")
    )
    print(f"Done. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()


