"""
Test DDP convergence with different initial guesses.

This script runs DDP/iLQR with 10 different initial control guesses
and compares the results.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path based on current file location
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ddp.src.ddp_casadi import (
    load_robot_from_urdf,
    CasadiSpaceRobotDynamics,
    CasadiRunningCost,
    CasadiTerminalCost,
    CasadiDDP,
)


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to quaternion [x, y, z, w].
    roll, pitch, yaw are in radians (floats).
    """
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q) + 1e-8
    return q


def quat_to_rot(q):
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix (NumPy).
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])


def geodesic_distance_so3(R1, R2):
    """
    Compute geodesic distance between two rotation matrices.
    theta = arccos((trace(R1^T @ R2) - 1) / 2)
    """
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    val = (trace - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)


def generate_initial_guesses(T: int, n_u: int, n_guesses: int = 10, seed: int = 42):
    """
    Generate different initial control guesses.
    
    Args:
        T: Time horizon (number of control steps)
        n_u: Control dimension
        n_guesses: Number of different initial guesses to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of (name, U0) tuples
    """
    np.random.seed(seed)
    guesses = []
    
    # 1. Zero initial guess
    guesses.append(("zeros", np.zeros((T, n_u))))
    
    # 2. Small constant positive
    guesses.append(("const_0.1", 0.1 * np.ones((T, n_u))))
    
    # 3. Small constant negative
    guesses.append(("const_-0.1", -0.1 * np.ones((T, n_u))))
    
    # 4. Random uniform small
    guesses.append(("random_small", 0.1 * np.random.uniform(-1, 1, (T, n_u))))
    
    # 5. Random uniform medium
    guesses.append(("random_medium", 0.5 * np.random.uniform(-1, 1, (T, n_u))))
    
    # 6. Random Gaussian small
    guesses.append(("gaussian_small", 0.1 * np.random.randn(T, n_u)))
    
    # 7. Random Gaussian medium
    guesses.append(("gaussian_medium", 0.5 * np.random.randn(T, n_u)))
    
    # 8. Sinusoidal pattern
    t_arr = np.linspace(0, 2 * np.pi, T)
    sin_guess = np.zeros((T, n_u))
    for j in range(n_u):
        sin_guess[:, j] = 0.2 * np.sin(t_arr + j * np.pi / n_u)
    guesses.append(("sinusoidal", sin_guess))
    
    # 9. Linear ramp up
    ramp_up = np.zeros((T, n_u))
    for t in range(T):
        ramp_up[t, :] = 0.3 * (t / T - 0.5)
    guesses.append(("ramp_up", ramp_up))
    
    # 10. Impulse at start
    impulse = np.zeros((T, n_u))
    impulse[:5, :] = 0.5  # First 5 timesteps have impulse
    guesses.append(("impulse_start", impulse))
    
    return guesses


def run_ddp_with_initial_guess(
    dyn, 
    running_cost, 
    terminal_cost, 
    x0, 
    U0, 
    dt, 
    max_iter=100,
    use_alm=False,
):
    """
    Run DDP with a given initial guess.
    
    Returns:
        X_opt, U_opt, cost_history, elapsed_time
    """
    # Create fresh running cost to reset ALM state
    running_cost_fresh = CasadiRunningCost(
        R_weight=running_cost.R,
        n_u=running_cost.n_u,
        joint_limits=(running_cost.jl_lower, running_cost.jl_upper) if running_cost.has_joint_limits else None,
        mu_init=1.0,
        lambda_init=0.0,
    )
    
    solver = CasadiDDP(
        dynamics_model=dyn,
        running_cost=running_cost_fresh,
        terminal_cost=terminal_cost,
        max_iter=max_iter,
        tol=1e-4,
        use_full_ddp=False,  # Use iLQR for speed
    )
    
    start_time = time.time()
    
    if use_alm:
        X_opt, U_opt, cost_history = solver.solve_alm(
            x0, U0.copy(), dt,
            alm_max_iter=5,
            constraint_tol=1e-4,
            mu_increase_factor=10.0,
        )
    else:
        X_opt, U_opt, cost_history = solver.solve(x0, U0.copy(), dt)
    
    elapsed_time = time.time() - start_time
    
    return X_opt, U_opt, cost_history, elapsed_time


def main():
    # Setup paths
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    urdf_path = os.path.join(root_dir, "assets", "SC_ur10e.urdf")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load robot
    robot = load_robot_from_urdf(urdf_path)
    n_q = robot["n_q"]
    n_x = 2 * n_q + 4
    
    # Time horizon
    T = 100
    dt = 0.1
    
    print(f"Testing DDP with different initial guesses")
    print(f"Time horizon: {T * dt}s ({T} steps, dt={dt}s)")
    print(f"State dim: {n_x}, Control dim: {n_q}")
    print("=" * 60)
    
    # Dynamics
    dyn = CasadiSpaceRobotDynamics(robot)
    
    # Initial state
    q0 = np.zeros(n_q)
    qd0 = np.zeros(n_q)
    q_base0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    x0 = np.concatenate([q0, qd0, q_base0])
    
    # Goal
    roll_deg, pitch_deg, yaw_deg = 150.0, 150.0, -15.0
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    q_goal = euler_to_quaternion(roll, pitch, yaw)
    goal_joints = np.zeros(n_q)
    goal_R = quat_to_rot(q_goal)
    
    print(f"Goal orientation: Roll={roll_deg}°, Pitch={pitch_deg}°, Yaw={yaw_deg}°")
    
    # Joint limits
    joint_limits = None
    if 'joints' in robot:
        moving_joints = [j for j in robot['joints'] if j['q_id'] != -1]
        moving_joints.sort(key=lambda x: x['q_id'])
        if len(moving_joints) == n_q:
            jl_lower = np.array([j['limit']['lower'] for j in moving_joints])
            jl_upper = np.array([j['limit']['upper'] for j in moving_joints])
            joint_limits = (jl_lower, jl_upper)
    
    # Costs
    running_cost = CasadiRunningCost(
        R_weight=0.01,
        n_u=n_q,
        joint_limits=joint_limits,
        mu_init=1.0,
        lambda_init=0.0,
    )
    
    terminal_cost = CasadiTerminalCost(
        goal_quaternion=q_goal,
        goal_joints=goal_joints,
        orientation_weight=20.0,
        joint_weight=1.0,
        joint_vel_weight=1.0,
        n_u=n_q,
    )
    
    # Generate initial guesses
    initial_guesses = generate_initial_guesses(T, n_q, n_guesses=10)
    
    # Results storage
    results = []
    
    # Run DDP with each initial guess
    for i, (name, U0) in enumerate(initial_guesses):
        print(f"\n{'='*60}")
        print(f"[{i+1}/10] Initial Guess: {name}")
        print(f"{'='*60}")
        
        X_opt, U_opt, cost_history, elapsed_time = run_ddp_with_initial_guess(
            dyn, running_cost, terminal_cost, x0, U0, dt,
            max_iter=100,
            use_alm=False,
        )
        
        # Compute final errors
        final_q_base = X_opt[-1, 2*n_q:]
        final_q_base /= np.linalg.norm(final_q_base) + 1e-8
        final_R = quat_to_rot(final_q_base)
        orient_err_rad = geodesic_distance_so3(final_R, goal_R)
        orient_err_deg = np.rad2deg(orient_err_rad)
        
        final_joints = X_opt[-1, :n_q]
        joint_err_norm = np.linalg.norm(final_joints - goal_joints)
        
        final_cost = cost_history[-1]
        n_iters = len(cost_history) - 1
        
        results.append({
            "name": name,
            "X_opt": X_opt,
            "U_opt": U_opt,
            "cost_history": cost_history,
            "elapsed_time": elapsed_time,
            "orient_err_deg": orient_err_deg,
            "joint_err_norm": joint_err_norm,
            "final_cost": final_cost,
            "n_iters": n_iters,
        })
        
        print(f"  Final cost: {final_cost:.4f}")
        print(f"  Orient error: {orient_err_deg:.2f}°")
        print(f"  Joint error: {joint_err_norm:.4f} rad")
        print(f"  Iterations: {n_iters}")
        print(f"  Time: {elapsed_time:.2f}s")
    
    # =========================================================================
    # Plotting
    # =========================================================================
    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)
    
    # Color palette for 10 guesses
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # --------------------------------------------------------------------------
    # Figure 1: Cost History Comparison
    # --------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    for i, res in enumerate(results):
        ax1.plot(res["cost_history"], label=res["name"], color=colors[i], linewidth=1.5)
    
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Cost (log scale)", fontsize=12)
    ax1.set_title("Cost Convergence with Different Initial Guesses", fontsize=14)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(results_dir, "initial_guess_cost_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    
    # --------------------------------------------------------------------------
    # Figure 2: Final Metrics Bar Chart
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r["name"] for r in results]
    final_costs = [r["final_cost"] for r in results]
    orient_errs = [r["orient_err_deg"] for r in results]
    times = [r["elapsed_time"] for r in results]
    
    x_pos = np.arange(len(names))
    
    # Final Cost
    axes2[0].bar(x_pos, final_costs, color=colors)
    axes2[0].set_ylabel("Final Cost")
    axes2[0].set_title("Final Cost by Initial Guess")
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes2[0].grid(True, alpha=0.3, axis="y")
    
    # Orientation Error
    axes2[1].bar(x_pos, orient_errs, color=colors)
    axes2[1].set_ylabel("Orientation Error (deg)")
    axes2[1].set_title("Final Orientation Error")
    axes2[1].set_xticks(x_pos)
    axes2[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes2[1].grid(True, alpha=0.3, axis="y")
    
    # Computation Time
    axes2[2].bar(x_pos, times, color=colors)
    axes2[2].set_ylabel("Time (s)")
    axes2[2].set_title("Computation Time")
    axes2[2].set_xticks(x_pos)
    axes2[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes2[2].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    fig2.savefig(os.path.join(results_dir, "initial_guess_metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    
    # --------------------------------------------------------------------------
    # Figure 3: Joint Trajectories Comparison (subset)
    # --------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
    time_steps = np.arange(T + 1) * dt
    
    # Pick 6 diverse initial guesses to show
    indices_to_show = [0, 3, 4, 6, 7, 9]  # zeros, random_small, random_medium, gaussian_medium, sinusoidal, impulse_start
    
    for ax_idx, res_idx in enumerate(indices_to_show):
        res = results[res_idx]
        ax = axes3.flatten()[ax_idx]
        
        # Plot all joint angles
        for j in range(n_q):
            ax.plot(time_steps, np.rad2deg(res["X_opt"][:, j]), label=f"J{j+1}")
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(f"Initial: {res['name']}")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(results_dir, "initial_guess_joint_trajectories.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    
    # --------------------------------------------------------------------------
    # Figure 4: Base Orientation Trajectories Comparison
    # --------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
    
    for ax_idx, res_idx in enumerate(indices_to_show):
        res = results[res_idx]
        ax = axes4.flatten()[ax_idx]
        
        # Quaternion components
        quats = res["X_opt"][:, 2*n_q:]
        ax.plot(time_steps, quats[:, 0], label="qx")
        ax.plot(time_steps, quats[:, 1], label="qy")
        ax.plot(time_steps, quats[:, 2], label="qz")
        ax.plot(time_steps, quats[:, 3], label="qw")
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Quaternion")
        ax.set_title(f"Initial: {res['name']}")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    fig4.savefig(os.path.join(results_dir, "initial_guess_orientation_trajectories.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)
    
    # --------------------------------------------------------------------------
    # Print Summary Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Name':<20} {'Final Cost':>12} {'Orient Err (°)':>15} {'Joint Err':>12} {'Iters':>8} {'Time (s)':>10}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['name']:<20} {res['final_cost']:>12.4f} {res['orient_err_deg']:>15.2f} "
              f"{res['joint_err_norm']:>12.4f} {res['n_iters']:>8} {res['elapsed_time']:>10.2f}")
    
    print("-" * 80)
    
    # Best result
    best_idx = np.argmin([r["final_cost"] for r in results])
    print(f"\nBest initial guess: {results[best_idx]['name']} (Final Cost: {results[best_idx]['final_cost']:.4f})")
    
    print(f"\nPlots saved to: {results_dir}")
    print("  - initial_guess_cost_comparison.png")
    print("  - initial_guess_metrics_comparison.png")
    print("  - initial_guess_joint_trajectories.png")
    print("  - initial_guess_orientation_trajectories.png")


if __name__ == "__main__":
    main()

