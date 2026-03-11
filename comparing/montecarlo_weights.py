"""
Monte Carlo sweep over DDP terminal cost weights.
Vary (orientation_weight, joint_weight, joint_vel_weight) and record
orientation error and joint error for each combination.
"""
import os
import sys
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ddp.src.ddp_casadi import (
    load_robot_from_urdf,
    CasadiSpaceRobotDynamics,
    CasadiRunningCost,
    CasadiTerminalCost,
    CasadiDDP,
)
from scenario import SCENARIO, get_goal_quaternion, get_initial_state
import src.dynamics.spart_casadi as spart_ca

SAVE_DIR = os.path.join(os.path.dirname(__file__), "montecarlo")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Weight grid ──
ORIENT_WEIGHTS = [1.0, 10.0, 50.0, 100.0, 500.0]
JOINT_WEIGHTS = [0.0, 1.0, 10.0, 50.0, 100.0]
JOINT_VEL_WEIGHTS = [0.0, 1.0, 10.0, 50.0, 100.0]


def run_single(dyn, robot, joint_limits, x0, q_goal, goal_joints,
               orient_w, joint_w, joint_vel_w):
    """Run DDP with given weights, return metrics dict."""
    n_q = robot["n_q"]
    T = SCENARIO["T"]
    dt = SCENARIO["dt"]

    running_cost = CasadiRunningCost(
        R_weight=SCENARIO["R_weight"],
        n_u=n_q,
        joint_limits=joint_limits,
        mu_init=1.0,
        lambda_init=0.0,
    )
    terminal_cost = CasadiTerminalCost(
        goal_quaternion=q_goal,
        goal_joints=goal_joints,
        orientation_weight=orient_w,
        joint_weight=joint_w,
        joint_vel_weight=joint_vel_w,
        vel_idx11_weight=SCENARIO["vel_idx11_weight"],
        n_u=n_q,
    )
    solver = CasadiDDP(
        dynamics_model=dyn,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        max_iter=200,
        tol=1e-4,
        use_full_ddp=False,
    )

    U0 = np.zeros((T, n_q))
    t0 = time.time()
    try:
        # Suppress DDP solver output
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            X_opt, U_opt, cost_history = solver.solve_alm(
                x0, U0, dt,
                alm_max_iter=5,
                constraint_tol=1e-4,
                mu_increase_factor=10.0,
            )
    except Exception as e:
        return {"converged": False, "error": str(e)}
    elapsed = time.time() - t0

    # Orientation error
    q_base_final = X_opt[-1, 2*n_q:]
    q_base_final = q_base_final / (np.linalg.norm(q_base_final) + 1e-8)
    R_final = np.array(spart_ca.quat_dcm(q_base_final))
    R_goal = np.array(spart_ca.quat_dcm(q_goal))
    R_diff = R_final - R_goal
    orient_trace = 0.5 * np.trace(R_diff.T @ R_diff)
    orient_cost = float(np.log(1e-8 + orient_trace))

    # Orientation angle (geodesic deg)
    trace_RtR = np.trace(R_final.T @ R_goal)
    cos_theta = np.clip((trace_RtR - 1) / 2, -1, 1)
    orient_angle_deg = float(np.degrees(np.arccos(cos_theta)))

    # Euler angles (deg) for debugging
    def _dcm_euler(R):
        pitch = np.arcsin(-np.clip(float(R[2, 0]), -1, 1))
        if np.abs(np.cos(pitch)) > 1e-6:
            yaw = np.arctan2(float(R[1, 0]), float(R[0, 0]))
            roll = np.arctan2(float(R[2, 1]), float(R[2, 2]))
        else:
            yaw = np.arctan2(-float(R[0, 1]), float(R[1, 1]))
            roll = 0.0
        return np.degrees([roll, pitch, yaw])
    euler_final = _dcm_euler(R_final)
    euler_goal = _dcm_euler(R_goal)

    # Joint error
    q_final = X_opt[-1, :n_q]
    joint_err = q_final - goal_joints
    joint_err_norm = float(np.linalg.norm(joint_err))
    joint_err_max_deg = float(np.degrees(np.max(np.abs(joint_err))))

    # Joint velocity at end
    qd_final = X_opt[-1, n_q:2*n_q]
    qd_norm = float(np.linalg.norm(qd_final))

    return {
        "converged": True,
        "orient_w": orient_w,
        "joint_w": joint_w,
        "joint_vel_w": joint_vel_w,
        "orient_cost": orient_cost,
        "orient_trace": orient_trace,
        "orient_angle_deg": orient_angle_deg,
        "euler_final": euler_final.tolist(),
        "euler_goal": euler_goal.tolist(),
        "joint_err_norm": joint_err_norm,
        "joint_err_max_deg": joint_err_max_deg,
        "qd_final_norm": qd_norm,
        "elapsed_s": elapsed,
        "final_joints": q_final.tolist(),
    }


def main():
    # Setup (once)
    urdf_path = os.path.join(ROOT_DIR, SCENARIO["urdf"])
    robot = load_robot_from_urdf(urdf_path)
    n_q = robot["n_q"]
    dyn = CasadiSpaceRobotDynamics(robot)

    joint_limits = None
    if 'joints' in robot:
        moving = sorted(
            [j for j in robot['joints'] if j['q_id'] != -1],
            key=lambda x: x['q_id']
        )
        if len(moving) == n_q:
            jl_lower = np.array([j['limit']['lower'] for j in moving])
            jl_upper = np.array([j['limit']['upper'] for j in moving])
            joint_limits = (jl_lower, jl_upper)

    x0 = get_initial_state()
    q_goal = get_goal_quaternion()
    goal_joints = np.array(SCENARIO["goal_joints"])

    combos = list(itertools.product(ORIENT_WEIGHTS, JOINT_WEIGHTS, JOINT_VEL_WEIGHTS))
    total = len(combos)
    print(f"=== Monte Carlo Weight Sweep ===")
    print(f"orient_w: {ORIENT_WEIGHTS}")
    print(f"joint_w:  {JOINT_WEIGHTS}")
    print(f"joint_vel_w: {JOINT_VEL_WEIGHTS}")
    print(f"Total combinations: {total}")
    print()

    results = []
    for idx, (ow, jw, jvw) in enumerate(combos):
        tag = f"[{idx+1}/{total}] ow={ow:6.1f} jw={jw:6.1f} jvw={jvw:6.1f}"
        print(f"{tag} ... ", end="", flush=True)
        res = run_single(dyn, robot, joint_limits, x0, q_goal, goal_joints,
                         ow, jw, jvw)
        if res["converged"]:
            ef = res['euler_final']
            print(f"orient={res['orient_angle_deg']:7.3f}deg  "
                  f"euler=[{ef[0]:+.1f},{ef[1]:+.1f},{ef[2]:+.1f}]  "
                  f"joint_max={res['joint_err_max_deg']:7.3f}deg  "
                  f"qd={res['qd_final_norm']:6.4f}  "
                  f"({res['elapsed_s']:.1f}s)")
        else:
            print(f"FAILED: {res['error']}")
        results.append(res)

    # ── Save CSV ──
    csv_path = os.path.join(SAVE_DIR, "weight_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("orient_w,joint_w,joint_vel_w,"
                "orient_angle_deg,orient_cost,orient_trace,"
                "joint_err_max_deg,joint_err_norm,qd_final_norm,"
                "elapsed_s,converged\n")
        for r in results:
            if r["converged"]:
                f.write(f"{r['orient_w']},{r['joint_w']},{r['joint_vel_w']},"
                        f"{r['orient_angle_deg']:.6f},{r['orient_cost']:.6f},{r['orient_trace']:.10f},"
                        f"{r['joint_err_max_deg']:.6f},{r['joint_err_norm']:.6f},{r['qd_final_norm']:.6f},"
                        f"{r['elapsed_s']:.2f},True\n")
            else:
                f.write(f"{r.get('orient_w','')},{r.get('joint_w','')},{r.get('joint_vel_w','')},"
                        f",,,,,,,"
                        f"False\n")
    print(f"\nCSV saved: {csv_path}")

    # ── Plots ──
    conv = [r for r in results if r["converged"]]
    if not conv:
        print("No converged results to plot.")
        return

    orient_angles = np.array([r["orient_angle_deg"] for r in conv])
    joint_errs = np.array([r["joint_err_max_deg"] for r in conv])
    ow_arr = np.array([r["orient_w"] for r in conv])
    jw_arr = np.array([r["joint_w"] for r in conv])
    jvw_arr = np.array([r["joint_vel_w"] for r in conv])

    # ── Plot 1: Orient error vs Joint error (Pareto front) ──
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(orient_angles, joint_errs,
                    c=np.log10(jw_arr + 0.1), cmap='viridis',
                    s=50, alpha=0.7, edgecolors='k', linewidth=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log10(joint_weight + 0.1)")
    ax.set_xlabel("Orientation Error (deg)")
    ax.set_ylabel("Max Joint Error (deg)")
    ax.set_title("DDP: Orientation vs Joint Error (Weight Sweep)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "pareto_orient_vs_joint.png"), dpi=150)
    print(f"Saved: {SAVE_DIR}/pareto_orient_vs_joint.png")

    # ── Plot 2: Heatmap for fixed joint_vel_w values ──
    unique_jvw = sorted(set(jvw_arr))
    n_panels = min(len(unique_jvw), 5)
    fig2, axes2 = plt.subplots(3, n_panels, figsize=(5*n_panels, 13))
    if n_panels == 1:
        axes2 = axes2.reshape(3, 1)

    for pi, jvw_val in enumerate(unique_jvw[:n_panels]):
        mask = jvw_arr == jvw_val
        sub = [r for r, m in zip(conv, mask) if m]
        if not sub:
            continue

        ow_vals = sorted(set(r["orient_w"] for r in sub))
        jw_vals = sorted(set(r["joint_w"] for r in sub))

        orient_grid = np.full((len(jw_vals), len(ow_vals)), np.nan)
        joint_grid = np.full((len(jw_vals), len(ow_vals)), np.nan)
        jvel_grid = np.full((len(jw_vals), len(ow_vals)), np.nan)

        for r in sub:
            oi = ow_vals.index(r["orient_w"])
            ji = jw_vals.index(r["joint_w"])
            orient_grid[ji, oi] = r["orient_angle_deg"]
            joint_grid[ji, oi] = r["joint_err_max_deg"]
            jvel_grid[ji, oi] = r["qd_final_norm"]

        # Orient heatmap
        ax = axes2[0, pi]
        im = ax.imshow(orient_grid, aspect='auto', origin='lower',
                       cmap='RdYlGn_r')
        ax.set_xticks(range(len(ow_vals)))
        ax.set_xticklabels([str(v) for v in ow_vals], fontsize=8)
        ax.set_yticks(range(len(jw_vals)))
        ax.set_yticklabels([str(v) for v in jw_vals], fontsize=8)
        ax.set_xlabel("orient_w")
        ax.set_ylabel("joint_w")
        ax.set_title(f"Orient err (deg)\njoint_vel_w={jvw_val}")
        for yi in range(len(jw_vals)):
            for xi in range(len(ow_vals)):
                v = orient_grid[yi, xi]
                if np.isfinite(v):
                    fmt = f"{v:.2f}" if v < 1.0 else f"{v:.1f}"
                    ax.text(xi, yi, fmt, ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Joint heatmap
        ax = axes2[1, pi]
        im = ax.imshow(joint_grid, aspect='auto', origin='lower',
                       cmap='RdYlGn_r')
        ax.set_xticks(range(len(ow_vals)))
        ax.set_xticklabels([str(v) for v in ow_vals], fontsize=8)
        ax.set_yticks(range(len(jw_vals)))
        ax.set_yticklabels([str(v) for v in jw_vals], fontsize=8)
        ax.set_xlabel("orient_w")
        ax.set_ylabel("joint_w")
        ax.set_title(f"Joint max err (deg)\njoint_vel_w={jvw_val}")
        for yi in range(len(jw_vals)):
            for xi in range(len(ow_vals)):
                v = joint_grid[yi, xi]
                if np.isfinite(v):
                    ax.text(xi, yi, f"{v:.1f}", ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Joint velocity heatmap
        ax = axes2[2, pi]
        im = ax.imshow(jvel_grid, aspect='auto', origin='lower',
                       cmap='RdYlGn_r')
        ax.set_xticks(range(len(ow_vals)))
        ax.set_xticklabels([str(v) for v in ow_vals], fontsize=8)
        ax.set_yticks(range(len(jw_vals)))
        ax.set_yticklabels([str(v) for v in jw_vals], fontsize=8)
        ax.set_xlabel("orient_w")
        ax.set_ylabel("joint_w")
        ax.set_title(f"Joint vel err (norm)\njoint_vel_w={jvw_val}")
        for yi in range(len(jw_vals)):
            for xi in range(len(ow_vals)):
                v = jvel_grid[yi, xi]
                if np.isfinite(v):
                    ax.text(xi, yi, f"{v:.3f}", ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig2.suptitle("DDP Weight Sweep: Heatmaps", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(SAVE_DIR, "heatmaps.png"), dpi=150)
    print(f"Saved: {SAVE_DIR}/heatmaps.png")

    # ── Plot 3: 1D slices ──
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

    # Slice: vary orient_w, fix joint_w=10, joint_vel_w=10
    sub = [r for r in conv if r["joint_w"] == 10.0 and r["joint_vel_w"] == 10.0]
    if sub:
        sub.sort(key=lambda r: r["orient_w"])
        xs = [r["orient_w"] for r in sub]
        ax = axes3[0]
        ax.plot(xs, [r["orient_angle_deg"] for r in sub], 'bo-', label="Orient err")
        ax.plot(xs, [r["joint_err_max_deg"] for r in sub], 'rs-', label="Joint max err")
        ax.set_xlabel("orientation_weight")
        ax.set_ylabel("Error (deg)")
        ax.set_title("Vary orient_w\n(joint_w=10, jvel_w=10)")
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Slice: vary joint_w, fix orient_w=100, joint_vel_w=10
    sub = [r for r in conv if r["orient_w"] == 100.0 and r["joint_vel_w"] == 10.0]
    if sub:
        sub.sort(key=lambda r: r["joint_w"])
        xs = [r["joint_w"] for r in sub]
        ax = axes3[1]
        ax.plot(xs, [r["orient_angle_deg"] for r in sub], 'bo-', label="Orient err")
        ax.plot(xs, [r["joint_err_max_deg"] for r in sub], 'rs-', label="Joint max err")
        ax.set_xlabel("joint_weight")
        ax.set_ylabel("Error (deg)")
        ax.set_title("Vary joint_w\n(orient_w=100, jvel_w=10)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Slice: vary joint_vel_w, fix orient_w=100, joint_w=10
    sub = [r for r in conv if r["orient_w"] == 100.0 and r["joint_w"] == 10.0]
    if sub:
        sub.sort(key=lambda r: r["joint_vel_w"])
        xs = [r["joint_vel_w"] for r in sub]
        ax = axes3[2]
        ax.plot(xs, [r["orient_angle_deg"] for r in sub], 'bo-', label="Orient err")
        ax.plot(xs, [r["joint_err_max_deg"] for r in sub], 'rs-', label="Joint max err")
        ax.set_xlabel("joint_vel_weight")
        ax.set_ylabel("Error (deg)")
        ax.set_title("Vary joint_vel_w\n(orient_w=100, joint_w=10)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig3.suptitle("DDP Weight Sweep: 1D Slices", fontsize=14)
    fig3.tight_layout()
    fig3.savefig(os.path.join(SAVE_DIR, "slices_1d.png"), dpi=150)
    print(f"Saved: {SAVE_DIR}/slices_1d.png")

    plt.close('all')
    print("\nDone.")


if __name__ == "__main__":
    main()
