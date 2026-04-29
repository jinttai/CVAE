"""
Compare optimize_torque.py (Newton + Null-space) vs
        optimize_torque_direct.py (Augmented Lagrangian).

Reads both results.npz files, regenerates trajectories/torques from their
optimized waypoints, and produces side-by-side comparison plots.
"""

import os
import sys
import math

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from src.utils.runtime_env import configure_windows_runtime

configure_windows_runtime()

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot


NUM_WAYPOINTS = 3
TOTAL_TIME = 10.0

NS_DIR = os.path.join(ROOT_DIR, "outputs/plots/torque_opt")
AL_DIR = os.path.join(ROOT_DIR, "outputs/plots/torque_direct")
OUT_DIR = os.path.join(ROOT_DIR, "outputs/plots/torque_compare")

METHOD_A = "Newton+Nullspace"
METHOD_B = "Augmented Lagrangian"
COLOR_A = "#4e79a7"
COLOR_B = "#e15759"


def euler_to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll / 2); sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2); sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2); sy = math.sin(yaw / 2)
    return torch.tensor([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def angle_error_deg(q_final, q_goal):
    dot = (q_final * q_goal).sum().abs().clamp(-1.0, 1.0)
    return float(2.0 * torch.acos(dot) * 180.0 / math.pi)


def simulate(physics, waypoints, q0_start, q0_goal):
    w = torch.as_tensor(waypoints, dtype=torch.float32, device=physics.device)
    if w.ndim == 1:
        w = w.unsqueeze(0)
    with torch.no_grad():
        q_traj, q_dot_traj = physics.generate_trajectory(w)
        tau = physics.compute_torques(q_traj[0], q_dot_traj[0])
        tc = physics.compute_torque_cost(q_traj[0], q_dot_traj[0]).item()
        _, q_final = physics.simulate_single(
            q_traj[0], q_dot_traj[0], q0_start, q0_goal
        )
    ae = angle_error_deg(q_final, q0_goal)
    return {
        "q": q_traj[0].cpu().numpy(),
        "qd": q_dot_traj[0].cpu().numpy(),
        "tau": tau.cpu().numpy(),
        "torque_cost": tc,
        "angle_err": ae,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"),
                          verbose_flag=False, device=device)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    n_q = robot["n_q"]

    # Same goal as both scripts: [15, 15, -15] deg
    q0_goal = euler_to_quaternion(
        math.radians(15.0), math.radians(15.0), math.radians(-15.0)
    ).to(device)
    q0_start = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)

    # ── Load both results ──
    ns_npz = np.load(os.path.join(NS_DIR, "results.npz"))
    al_npz = np.load(os.path.join(AL_DIR, "results.npz"))

    w_ns_init = ns_npz["waypoints_init"]
    w_ns_opt = ns_npz["waypoints_optimized"]
    w_al_opt = al_npz["waypoints_optimized"]

    res_ns_init = simulate(physics, w_ns_init, q0_start, q0_goal)
    res_ns = simulate(physics, w_ns_opt, q0_start, q0_goal)
    res_al = simulate(physics, w_al_opt, q0_start, q0_goal)

    print(f"\n{'Method':<22s} | {'Torque Cost':>12s} | {'Angle Err':>10s}")
    print("-" * 52)
    print(f"{'CVAE init (NS)':<22s} | {res_ns_init['torque_cost']:12.2f} | "
          f"{res_ns_init['angle_err']:8.3f} deg")
    print(f"{METHOD_A:<22s} | {res_ns['torque_cost']:12.2f} | "
          f"{res_ns['angle_err']:8.3f} deg")
    print(f"{METHOD_B:<22s} | {res_al['torque_cost']:12.2f} | "
          f"{res_al['angle_err']:8.3f} deg")

    t_axis = np.linspace(0, TOTAL_TIME, res_ns["q"].shape[0])

    # ── Figure 1: Joint trajectory / velocity / torque overlay ──
    fig, axes = plt.subplots(3, n_q, figsize=(4 * n_q, 10), sharex=True)
    rows = [
        ("Joint Angle (rad)", "q"),
        ("Joint Velocity (rad/s)", "qd"),
        ("Torque (Nm)", "tau"),
    ]
    for row, (ylabel, key) in enumerate(rows):
        for j in range(n_q):
            ax = axes[row, j]
            ax.plot(t_axis, res_ns_init[key][:, j],
                    color="#888888", linewidth=1.0, alpha=0.6,
                    label="CVAE init", linestyle=":")
            ax.plot(t_axis, res_ns[key][:, j],
                    color=COLOR_A, linewidth=1.4, alpha=0.9, label=METHOD_A)
            ax.plot(t_axis, res_al[key][:, j],
                    color=COLOR_B, linewidth=1.4, alpha=0.9, label=METHOD_B)
            if row == 0:
                ax.set_title(f"J{j+1}", fontsize=11)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == 2:
                ax.set_xlabel("Time (s)", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Trajectory & Torque: Method Comparison", fontsize=13, y=1.03)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "trajectory_compare.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Torque norm over time ──
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    for res, name, c in [
        (res_ns_init, "CVAE init", "#888888"),
        (res_ns, METHOD_A, COLOR_A),
        (res_al, METHOD_B, COLOR_B),
    ]:
        tau_norm = np.linalg.norm(res["tau"], axis=1)
        ax2.plot(t_axis, tau_norm, label=f"{name} (cost={res['torque_cost']:.1f})",
                 color=c, linewidth=1.6, alpha=0.9)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("||torque|| (Nm)")
    ax2.set_title("Torque Norm Over Time")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(OUT_DIR, "torque_norm_compare.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    # ── Figure 3: Summary bar chart ──
    fig3, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5))

    names = ["CVAE init", METHOD_A, METHOD_B]
    tc_vals = [res_ns_init["torque_cost"], res_ns["torque_cost"], res_al["torque_cost"]]
    ae_vals = [res_ns_init["angle_err"], res_ns["angle_err"], res_al["angle_err"]]
    colors = ["#888888", COLOR_A, COLOR_B]

    bars = axA.bar(names, tc_vals, color=colors, edgecolor="white")
    axA.set_ylabel("Torque Cost")
    axA.set_title("Torque Cost Comparison")
    for bar, v in zip(bars, tc_vals):
        axA.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    axA.grid(True, axis="y", alpha=0.25)

    bars = axB.bar(names, ae_vals, color=colors, edgecolor="white")
    axB.set_ylabel("Angle Error (deg)")
    axB.set_title("Orientation Error Comparison")
    for bar, v in zip(bars, ae_vals):
        axB.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axB.grid(True, axis="y", alpha=0.25)

    fig3.suptitle("Method Comparison Summary", fontsize=13)
    fig3.tight_layout()
    p3 = os.path.join(OUT_DIR, "summary_compare.png")
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)

    # ── Figure 4: Convergence curves (each method's own history) ──
    # Newton+Nullspace: torque_history is in npz? No — it's only saved in print/info.
    # AL: torque_history is in results.npz
    fig4, ax4 = plt.subplots(figsize=(9, 4.5))
    if "torque_history" in al_npz.files:
        th = al_npz["torque_history"]
        ax4.plot(range(1, len(th) + 1), th, "o-",
                 color=COLOR_B, label=f"{METHOD_B} (outer)", linewidth=1.5, markersize=4)
    ax4.axhline(y=res_ns["torque_cost"], color=COLOR_A, linestyle="--",
                linewidth=1.5, label=f"{METHOD_A} final = {res_ns['torque_cost']:.1f}")
    ax4.axhline(y=res_ns_init["torque_cost"], color="#888888", linestyle=":",
                linewidth=1.2, label=f"CVAE init = {res_ns_init['torque_cost']:.1f}")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Torque Cost")
    ax4.set_title("Convergence: AL outer iterations vs Nullspace final")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    p4 = os.path.join(OUT_DIR, "convergence_compare.png")
    fig4.savefig(p4, dpi=150)
    plt.close(fig4)

    print(f"\nSaved:")
    for p in [p1, p2, p3, p4]:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
