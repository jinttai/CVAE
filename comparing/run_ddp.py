"""
Run DDP/iLQR for the shared SCENARIO and save results to comparing/results_ddp/.
"""
import os
import sys
import time
import numpy as np

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


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "results_ddp")
    os.makedirs(save_dir, exist_ok=True)

    # Load robot
    urdf_path = os.path.join(ROOT_DIR, SCENARIO["urdf"])
    robot = load_robot_from_urdf(urdf_path)
    n_q = robot["n_q"]

    # Scenario params
    T = SCENARIO["T"]
    dt = SCENARIO["dt"]
    total_time = SCENARIO["total_time"]
    x0 = get_initial_state()
    q_goal = get_goal_quaternion()
    goal_joints = np.array(SCENARIO["goal_joints"])

    print(f"=== DDP Solver (SCENARIO) ===")
    print(f"Time: {total_time}s ({T} steps, dt={dt}s)")
    print(f"Goal euler (deg): {SCENARIO['goal_euler_deg']}")
    print(f"Goal quat: {q_goal}")
    print(f"Weights: orient={SCENARIO['orientation_weight']}, joint={SCENARIO['joint_weight']}, "
          f"joint_vel={SCENARIO['joint_vel_weight']}, vel_idx11={SCENARIO['vel_idx11_weight']}, "
          f"R={SCENARIO['R_weight']}")

    # Dynamics
    dyn = CasadiSpaceRobotDynamics(robot)

    # Joint limits from URDF
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

    # Costs (from SCENARIO)
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
        orientation_weight=SCENARIO["orientation_weight"],
        joint_weight=SCENARIO["joint_weight"],
        joint_vel_weight=SCENARIO["joint_vel_weight"],
        vel_idx11_weight=SCENARIO["vel_idx11_weight"],
        n_u=n_q,
    )

    solver = CasadiDDP(
        dynamics_model=dyn,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        max_iter=500,
        tol=1e-4,
        use_full_ddp=False,  # iLQR
    )

    U0 = np.zeros((T, n_q))
    print("\nStarting iLQR + ALM solve...")
    t_start = time.time()
    X_opt, U_opt, cost_history = solver.solve_alm(
        x0, U0, dt,
        alm_max_iter=10,
        constraint_tol=1e-4,
        mu_increase_factor=10.0,
    )
    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.2f}s")

    # Extract final orientation
    q_base_final = X_opt[-1, 2*n_q:]
    q_base_final = q_base_final / (np.linalg.norm(q_base_final) + 1e-8)
    R_final = np.array(spart_ca.quat_dcm(q_base_final))
    R_goal = np.array(spart_ca.quat_dcm(q_goal))
    R_diff = R_final - R_goal
    orient_error = 0.5 * np.trace(R_diff.T @ R_diff)
    orient_cost = float(np.log(1e-8 + orient_error))

    # Geodesic angle (deg) — same metric as heatmap
    trace_RtR = np.trace(R_final.T @ R_goal)
    cos_theta = np.clip((trace_RtR - 1) / 2, -1, 1)
    orient_angle_deg = float(np.degrees(np.arccos(cos_theta)))

    # Final euler angles (deg)
    pitch_f = np.arcsin(-np.clip(R_final[2, 0], -1, 1))
    if np.abs(np.cos(pitch_f)) > 1e-6:
        yaw_f = np.arctan2(R_final[1, 0], R_final[0, 0])
        roll_f = np.arctan2(R_final[2, 1], R_final[2, 2])
    else:
        yaw_f = np.arctan2(-R_final[0, 1], R_final[1, 1])
        roll_f = 0.0
    euler_final_deg = np.degrees([roll_f, pitch_f, yaw_f])

    # Goal euler (deg)
    pitch_g = np.arcsin(-np.clip(R_goal[2, 0], -1, 1))
    if np.abs(np.cos(pitch_g)) > 1e-6:
        yaw_g = np.arctan2(R_goal[1, 0], R_goal[0, 0])
        roll_g = np.arctan2(R_goal[2, 1], R_goal[2, 2])
    else:
        yaw_g = np.arctan2(-R_goal[0, 1], R_goal[1, 1])
        roll_g = 0.0
    euler_goal_deg = np.degrees([roll_g, pitch_g, yaw_g])

    print(f"\nGoal  euler (deg): roll={euler_goal_deg[0]:+.4f}  pitch={euler_goal_deg[1]:+.4f}  yaw={euler_goal_deg[2]:+.4f}")
    print(f"Final euler (deg): roll={euler_final_deg[0]:+.4f}  pitch={euler_final_deg[1]:+.4f}  yaw={euler_final_deg[2]:+.4f}")
    print(f"Error euler (deg): roll={euler_final_deg[0]-euler_goal_deg[0]:+.4f}  pitch={euler_final_deg[1]-euler_goal_deg[1]:+.4f}  yaw={euler_final_deg[2]-euler_goal_deg[2]:+.4f}")
    print(f"Orient geodesic error: {orient_angle_deg:.4f} deg")
    print(f"Orient trace error: {orient_error:.8f}")
    print(f"Orient cost (log): {orient_cost:.8f}")

    # Save results
    t_vec = np.linspace(0, total_time, T + 1)
    q_joints = X_opt[:, :n_q]         # [T+1, 6]
    qd_joints = X_opt[:, n_q:2*n_q]  # [T+1, 6]

    # Joint trajectory CSV
    header_q = "t," + ",".join([f"J{i+1}" for i in range(n_q)])
    np.savetxt(
        os.path.join(save_dir, "q_traj.csv"),
        np.column_stack([t_vec, q_joints]),
        delimiter=",", header=header_q, comments="",
    )

    # Joint velocity CSV
    header_qd = "t," + ",".join([f"dJ{i+1}" for i in range(n_q)])
    np.savetxt(
        os.path.join(save_dir, "qd_traj.csv"),
        np.column_stack([t_vec, qd_joints]),
        delimiter=",", header=header_qd, comments="",
    )

    # Orientation trajectory (compute euler at each step)
    euler_traj = []
    for k in range(T + 1):
        qb = X_opt[k, 2*n_q:]
        qb = qb / (np.linalg.norm(qb) + 1e-8)
        R = np.array(spart_ca.quat_dcm(qb))
        # ZYX euler: yaw, pitch, roll
        pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
        if np.abs(np.cos(pitch)) > 1e-6:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            yaw = np.arctan2(-R[0, 1], R[1, 1])
            roll = 0.0
        euler_traj.append([yaw, pitch, roll])
    euler_traj = np.array(euler_traj)

    header_euler = "t,yaw,pitch,roll"
    np.savetxt(
        os.path.join(save_dir, "euler_traj.csv"),
        np.column_stack([t_vec, euler_traj]),
        delimiter=",", header=header_euler, comments="",
    )

    # Controls CSV
    t_ctrl = np.linspace(0, total_time - dt, T)
    header_u = "t," + ",".join([f"u{i+1}" for i in range(n_q)])
    np.savetxt(
        os.path.join(save_dir, "controls.csv"),
        np.column_stack([t_ctrl, U_opt]),
        delimiter=",", header=header_u, comments="",
    )

    # Meta
    with open(os.path.join(save_dir, "meta.csv"), "w") as f:
        f.write("key,value\n")
        f.write(f"elapsed_s,{elapsed:.4f}\n")
        f.write(f"orient_cost,{orient_cost:.8f}\n")
        f.write(f"orient_error,{orient_error:.8f}\n")
        f.write(f"total_time,{total_time}\n")
        f.write(f"dt,{dt}\n")
        f.write(f"T,{T}\n")

    np.save(os.path.join(save_dir, "X_opt.npy"), X_opt)
    np.save(os.path.join(save_dir, "U_opt.npy"), U_opt)
    np.save(os.path.join(save_dir, "cost_history.npy"), np.array(cost_history))

    print(f"\nResults saved to {save_dir}/")


if __name__ == "__main__":
    main()
