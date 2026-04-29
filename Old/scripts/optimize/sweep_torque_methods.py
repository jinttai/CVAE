"""
Sweep multiple goal orientations and compare:
  - Newton + Null-space L-BFGS (from scripts/optimize_torque.py)
  - Augmented Lagrangian       (from scripts/optimize/optimize_torque_direct.py)

Reuses the core routines of both scripts by importing them, so CVAE/physics
setup only happens once.
"""

import os
import sys
import time
import math
import importlib.util

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from src.utils.runtime_env import configure_windows_runtime

configure_windows_runtime()

import torch
import numpy as np
from torch.func import vmap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.physics_layer import PhysicsLayer
from src.dynamics.urdf2robot_torch import urdf2robot
from src.models.cvae import CVAE


NUM_WAYPOINTS = 3
TOTAL_TIME = 10.0
N_SAMPLES = 128
CVAE_WEIGHT_PATH = os.path.join(ROOT_DIR, "outputs/weights/cvae_debug/v5_joint_change.pth")
OUT_DIR = os.path.join(ROOT_DIR, "outputs/plots/torque_sweep")

# Goal orientations to test (roll, pitch, yaw in degrees)
GOAL_ORIENTATIONS_DEG = [
    (15.0, 15.0, -15.0),
    (30.0, 20.0, -45.0),
    (45.0, -30.0, 60.0),
    (-20.0, -40.0, 80.0),
    (0.0, 60.0, 0.0),
]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import both optimization routines
NS_PATH = os.path.join(ROOT_DIR, "scripts/optimize_torque.py")
AL_PATH = os.path.join(ROOT_DIR, "scripts/optimize/optimize_torque_direct.py")
ns_mod = _load_module(NS_PATH, "optimize_torque_ns")
al_mod = _load_module(AL_PATH, "optimize_torque_al")


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
    dot = (q_final * q_goal).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot) * 180.0 / math.pi


def pick_best_cvae(cvae, physics, q0_start, q0_goal, device, n_samples=N_SAMPLES):
    q0_start_b = q0_start.expand(n_samples, -1)
    q0_goal_b = q0_goal.expand(n_samples, -1)
    with torch.no_grad():
        cond = torch.cat([q0_start_b, q0_goal_b], dim=1)
        w_all = cvae.inference(cond)
        q_tr, qd_tr = physics.generate_trajectory(w_all)
        batch_sim = vmap(physics.simulate_single, in_dims=(0, 0, 0, 0))
        losses, qf = batch_sim(q_tr, qd_tr, q0_start_b, q0_goal_b)
    best = losses.argmin().item()
    return w_all, best, losses, qf


def simulate(physics, w, q0_start, q0_goal):
    if w.ndim == 1:
        w = w.unsqueeze(0)
    with torch.no_grad():
        q, qd = physics.generate_trajectory(w)
        tau = physics.compute_torques(q[0], qd[0])
        tc = physics.compute_torque_cost(q[0], qd[0]).item()
        _, qf = physics.simulate_single(q[0], qd[0], q0_start, q0_goal)
    ae = angle_error_deg(qf.unsqueeze(0), q0_goal.unsqueeze(0)).item()
    return {"q": q[0].cpu().numpy(), "qd": qd[0].cpu().numpy(),
            "tau": tau.cpu().numpy(), "tc": tc, "ae": ae}


def run_one(cvae, physics, roll_deg, pitch_deg, yaw_deg, device):
    q0_goal = euler_to_quaternion(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg),
    ).to(device)
    q0_start = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)

    w_all, best, losses, qf = pick_best_cvae(cvae, physics, q0_start, q0_goal, device)
    w_best = w_all[best]
    init_res = simulate(physics, w_best, q0_start, q0_goal)

    # ── Newton + Nullspace ──
    t0 = time.time()
    w_ns, info_ns = ns_mod.optimize_single_sample(
        physics, w_best, q0_start, q0_goal, device, verbose=False,
    )
    ns_time = time.time() - t0
    ns_res = simulate(physics, w_ns, q0_start, q0_goal)

    # ── Augmented Lagrangian ──
    t0 = time.time()
    al_out = al_mod.run_optimization(
        physics, w_best.unsqueeze(0),
        q0_start.unsqueeze(0), q0_goal.unsqueeze(0),
        None, label="AL", verbose=False,
    )
    al_time = time.time() - t0
    al_res = simulate(physics, al_out["waypoints"], q0_start, q0_goal)

    return {
        "goal_deg": (roll_deg, pitch_deg, yaw_deg),
        "cvae_best_idx": best,
        "init": init_res,
        "ns": ns_res,
        "al": al_res,
        "ns_time": ns_time,
        "al_time": al_time,
        "al_torque_history": list(al_out["torque_history"]),
        "al_orient_history": list(al_out["orient_history"]),
        "ns_torque_history": list(info_ns["torque_history"]),
    }


def plot_sweep(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = [f"[{r['goal_deg'][0]:.0f},{r['goal_deg'][1]:.0f},{r['goal_deg'][2]:.0f}]"
              for r in results]

    tc_init = [r["init"]["tc"] for r in results]
    tc_ns = [r["ns"]["tc"] for r in results]
    tc_al = [r["al"]["tc"] for r in results]

    ae_init = [r["init"]["ae"] for r in results]
    ae_ns = [r["ns"]["ae"] for r in results]
    ae_al = [r["al"]["ae"] for r in results]

    ns_time = [r["ns_time"] for r in results]
    al_time = [r["al_time"] for r in results]

    x = np.arange(len(labels))
    width = 0.28

    # ── Figure 1: Torque cost + angle error grouped bars ──
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    axA.bar(x - width, tc_init, width, color="#888888", label="CVAE init")
    axA.bar(x,         tc_ns,   width, color="#4e79a7", label="Newton+Nullspace")
    axA.bar(x + width, tc_al,   width, color="#e15759", label="AL")
    axA.set_yscale("log")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, rotation=20)
    axA.set_ylabel("Torque Cost (log)")
    axA.set_title("Torque Cost by Goal Orientation [r,p,y deg]")
    axA.legend(fontsize=9)
    axA.grid(True, axis="y", alpha=0.3, which="both")

    axB.bar(x - width, ae_init, width, color="#888888", label="CVAE init")
    axB.bar(x,         ae_ns,   width, color="#4e79a7", label="Newton+Nullspace")
    axB.bar(x + width, ae_al,   width, color="#e15759", label="AL")
    axB.axhline(y=al_mod.ORIENT_TOL_DEG, color="red", linestyle="--", alpha=0.6,
                label=f"AL tol={al_mod.ORIENT_TOL_DEG} deg")
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=20)
    axB.set_ylabel("Angle Error (deg)")
    axB.set_title("Orientation Error by Goal")
    axB.legend(fontsize=9)
    axB.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    p1 = os.path.join(out_dir, "sweep_summary.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ── Figure 2: Trade-off scatter (torque vs angle error) ──
    fig2, ax = plt.subplots(figsize=(8, 5.5))
    for xi, (tc, ae, lab) in enumerate(zip(tc_ns, ae_ns, labels)):
        ax.scatter(ae, tc, color="#4e79a7", s=80, zorder=3)
        ax.annotate(lab, (ae, tc), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color="#4e79a7")
    for xi, (tc, ae, lab) in enumerate(zip(tc_al, ae_al, labels)):
        ax.scatter(ae, tc, color="#e15759", s=80, marker="s", zorder=3)
        ax.annotate(lab, (ae, tc), xytext=(5, -10),
                    textcoords="offset points", fontsize=8, color="#e15759")
    ax.scatter([], [], color="#4e79a7", s=80, label="Newton+Nullspace")
    ax.scatter([], [], color="#e15759", s=80, marker="s", label="AL")
    ax.axvline(x=al_mod.ORIENT_TOL_DEG, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Angle Error (deg)")
    ax.set_ylabel("Torque Cost")
    ax.set_title("Trade-off: Torque Cost vs Orientation Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "sweep_tradeoff.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    # ── Figure 3: Runtime comparison ──
    fig3, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, ns_time, width, color="#4e79a7", label="Newton+Nullspace")
    ax.bar(x + width / 2, al_time, width, color="#e15759", label="AL")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Wall-clock Runtime per Goal")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout()
    p3 = os.path.join(out_dir, "sweep_runtime.png")
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)

    # ── Figure 4: Relative torque reduction vs CVAE init ──
    fig4, ax = plt.subplots(figsize=(9, 4.5))
    red_ns = [(1 - r["ns"]["tc"] / r["init"]["tc"]) * 100 for r in results]
    red_al = [(1 - r["al"]["tc"] / r["init"]["tc"]) * 100 for r in results]
    ax.bar(x - width / 2, red_ns, width, color="#4e79a7", label="Newton+Nullspace")
    ax.bar(x + width / 2, red_al, width, color="#e15759", label="AL")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Torque Reduction vs CVAE init (%)")
    ax.set_title("Relative Torque Reduction")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(red_ns):
        ax.text(i - width / 2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(red_al):
        ax.text(i + width / 2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    fig4.tight_layout()
    p4 = os.path.join(out_dir, "sweep_reduction.png")
    fig4.savefig(p4, dpi=150)
    plt.close(fig4)

    # ── Figure 5: Per-goal torque norm over time ──
    n_goals = len(results)
    ncol = min(3, n_goals)
    nrow = (n_goals + ncol - 1) // ncol
    fig5, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow), sharex=True)
    axes = np.atleast_2d(axes).reshape(nrow, ncol)
    t_axis = np.linspace(0, TOTAL_TIME, results[0]["init"]["tau"].shape[0])
    for idx, r in enumerate(results):
        ax = axes[idx // ncol, idx % ncol]
        for key, name, color in [("init", "CVAE init", "#888888"),
                                  ("ns", "Nullspace", "#4e79a7"),
                                  ("al", "AL", "#e15759")]:
            tn = np.linalg.norm(r[key]["tau"], axis=1)
            ax.plot(t_axis, tn, color=color, label=f"{name} ({r[key]['tc']:.0f})",
                    linewidth=1.4)
        ax.set_title(labels[idx], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if idx // ncol == nrow - 1:
            ax.set_xlabel("Time (s)")
        if idx % ncol == 0:
            ax.set_ylabel("||τ|| (Nm)")
    # hide unused
    for idx in range(n_goals, nrow * ncol):
        axes[idx // ncol, idx % ncol].axis("off")
    fig5.suptitle("Torque Norm Over Time (per goal)", fontsize=12, y=1.0)
    fig5.tight_layout()
    p5 = os.path.join(out_dir, "sweep_torque_norm.png")
    fig5.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig5)

    return [p1, p2, p3, p4, p5]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    robot, _ = urdf2robot(os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf"),
                          verbose_flag=False, device=device)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    n_q = robot["n_q"]
    output_dim = NUM_WAYPOINTS * n_q

    cvae = CVAE(8, output_dim, 3, joint_limits=robot["joint_limits"]).to(device)
    cvae.load_state_dict(torch.load(CVAE_WEIGHT_PATH, map_location=device,
                                    weights_only=True))
    cvae.eval()
    print(f"Loaded CVAE from {CVAE_WEIGHT_PATH}")

    torch.manual_seed(42)

    results = []
    for i, (r, p, y) in enumerate(GOAL_ORIENTATIONS_DEG):
        print(f"\n{'='*70}")
        print(f"Goal {i+1}/{len(GOAL_ORIENTATIONS_DEG)}: roll={r}, pitch={p}, yaw={y} deg")
        print(f"{'='*70}")
        res = run_one(cvae, physics, r, p, y, device)
        print(f"  CVAE init        : tc={res['init']['tc']:10.1f}  ae={res['init']['ae']:.3f}°")
        print(f"  Newton+Nullspace : tc={res['ns']['tc']:10.1f}  ae={res['ns']['ae']:.3f}°  "
              f"({res['ns_time']:.1f}s)")
        print(f"  Augmented Lagr.  : tc={res['al']['tc']:10.1f}  ae={res['al']['ae']:.3f}°  "
              f"({res['al_time']:.1f}s)")
        results.append(res)

    # ── Summary table ──
    print(f"\n{'='*80}")
    print(f"{'Goal (r,p,y)':<18s} | {'CVAE init':>12s} | "
          f"{'Nullspace':>12s} | {'AL':>12s} | "
          f"{'NS ae°':>8s} | {'AL ae°':>8s}")
    print("-" * 80)
    for r in results:
        g = r["goal_deg"]
        gl = f"[{g[0]:+.0f},{g[1]:+.0f},{g[2]:+.0f}]"
        print(f"{gl:<18s} | {r['init']['tc']:12.1f} | "
              f"{r['ns']['tc']:12.1f} | {r['al']['tc']:12.1f} | "
              f"{r['ns']['ae']:8.3f} | {r['al']['ae']:8.3f}")

    # ── Plot ──
    paths = plot_sweep(results, OUT_DIR)
    print(f"\nPlots saved to {OUT_DIR}:")
    for p in paths:
        print(f"  -> {p}")

    # ── Save raw data ──
    save_data = {
        "goals_deg": np.array(GOAL_ORIENTATIONS_DEG),
        "init_tc": np.array([r["init"]["tc"] for r in results]),
        "ns_tc":   np.array([r["ns"]["tc"] for r in results]),
        "al_tc":   np.array([r["al"]["tc"] for r in results]),
        "init_ae": np.array([r["init"]["ae"] for r in results]),
        "ns_ae":   np.array([r["ns"]["ae"] for r in results]),
        "al_ae":   np.array([r["al"]["ae"] for r in results]),
        "ns_time": np.array([r["ns_time"] for r in results]),
        "al_time": np.array([r["al_time"] for r in results]),
    }
    np.savez(os.path.join(OUT_DIR, "sweep_results.npz"), **save_data)
    print(f"  -> {os.path.join(OUT_DIR, 'sweep_results.npz')}")


if __name__ == "__main__":
    main()
