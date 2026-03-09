"""
Convert IPOPT results (state + acceleration control) to joint torques.

Loads saved trajectory (trajectory_casadi_ipopt_states.npy, trajectory_casadi_ipopt_controls.npy),
uses SPART inverse dynamics to compute tau at each step, then plots and saves.
Same structure as ddp_acc/scripts/result_to_torque.py.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ddp.src.ddp_casadi import load_robot_from_urdf, CasadiSpaceRobotDynamics


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    urdf_path = os.path.join(root_dir, "assets", "SC_ur10e.urdf")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

    robot = load_robot_from_urdf(urdf_path)
    n_q = robot["n_q"]
    dyn = CasadiSpaceRobotDynamics(robot)

    states_path = os.path.join(results_dir, "trajectory_casadi_ipopt_states.npy")
    controls_path = os.path.join(results_dir, "trajectory_casadi_ipopt_controls.npy")

    if not os.path.isfile(states_path) or not os.path.isfile(controls_path):
        print("Saved trajectory not found. Run run_ipopt.py first.")
        print(f"  Expected: {states_path}")
        print(f"            {controls_path}")
        sys.exit(1)

    X_opt = np.load(states_path)
    U_opt = np.load(controls_path)

    T = U_opt.shape[0]
    dt = 0.1
    time_steps = np.arange(T) * dt

    print("Computing joint torques from inverse dynamics...")
    Tau = np.zeros((T, n_q))
    for t in range(T):
        q = X_opt[t, :n_q]
        qd = X_opt[t, n_q : 2 * n_q]
        qdd = U_opt[t]
        Tau[t] = dyn.compute_torque(q, qd, qdd)

    print(f"Torque trajectory shape: {Tau.shape}")
    print(f"Torque range (N·m): min={Tau.min():.4f}, max={Tau.max():.4f}")

    tau_npy_path = os.path.join(results_dir, "trajectory_casadi_ipopt_torques.npy")
    np.save(tau_npy_path, Tau)
    print(f"Saved: {tau_npy_path}")

    tau_csv_path = os.path.join(results_dir, "trajectory_casadi_ipopt_torques.csv")
    header = "time," + ",".join([f"tau_{i+1}" for i in range(n_q)])
    np.savetxt(
        tau_csv_path,
        np.hstack([time_steps.reshape(-1, 1), Tau]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6f",
    )
    print(f"Saved: {tau_csv_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_q):
        ax.plot(time_steps, Tau[:, i], label=f"Joint {i+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title("Joint torques (from IPOPT acceleration result)")
    ax.legend(loc="upper right", ncol=2, fontsize="small")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "trajectory_casadi_ipopt_torques.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
