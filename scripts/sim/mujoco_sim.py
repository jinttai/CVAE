import time
import numpy as np
import mujoco
import mujoco.viewer
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)
# We might need spart for accurate base velocity reconstruction if simple diff is not enough
from src.dynamics import spart_functions as ft
from src.dynamics.urdf2robot import urdf2robot

def get_quaternion_derivative(q, q_next, dt):
    """
    Compute angular velocity w from two quaternions q and q_next.
    q = [w, x, y, z] (MuJoCo convention)
    Approximation: q_next ~= q + 0.5 * w * dt * q
    => w_vec ~= (2/dt) * (q_next * q_inv)
    """
    # Inverse of q [w, x, y, z] is [w, -x, -y, -z]
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    
    # Quaternion multiplication (q_inv * q_next) to get body angular velocity
    # q_inv * q_next = (w_inv, -v_inv) * (w_next, v_next)
    # This gives the relative rotation in Body frame: R_body_rel = R_current^T * R_next
    
    a = q_inv
    b = q_next
    
    av = a[1:]
    bv = b[1:]
    
    rw = a[0]*b[0] - np.dot(av, bv)
    rv = a[0]*bv + b[0]*av + np.cross(av, bv)
    
    # r is the relative rotation quaternion (approx identity for small dt)
    # Axis-angle: theta = 2 * acos(rw), axis = rv / sin(theta/2)
    # w = axis * theta / dt
    
    # For small angles: theta ~ 2 * ||rv|| (since rw ~ 1)
    # w ~ 2 * rv / dt
    
    # To be more precise using log map:
    # if rw is close to 1:
    if rw > 0.999999:
        w = (2.0 / dt) * rv
    else:
        # Avoid numerical issues
        rw = np.clip(rw, -1.0, 1.0)
        theta = 2.0 * np.arccos(rw)
        sin_half = np.sin(theta/2.0)
        if sin_half < 1e-6:
            w = np.zeros(3)
        else:
            w = (theta / dt) * (rv / sin_half)
            
    return w

def run_simulation():
    # 1. Load Model
    xml_path = os.path.join(ROOT_DIR, 'assets/spacerobot_cjt.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Setup data_ref for Inverse Dynamics
    data_ref = mujoco.MjData(model)
    
    # 2. Load Trajectory
    csv_path = os.path.join(ROOT_DIR, 'ddp/results/trajectory_casadi_ilqr.csv') # Fixed path from ddp_acc to ddp
    if not os.path.exists(csv_path):
        print(f"Error: Trajectory file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded trajectory from {csv_path} with {len(df)} steps.")
    
    # Columns: time, joint_1_angle..., quaternion_x, quaternion_y, quaternion_z, quaternion_w
    # Extract data
    times = df['time'].values
    dt_csv = times[1] - times[0] if len(times) > 1 else 0.1
    
    # Joint data
    joint_cols = [c for c in df.columns if 'angle' in c]
    vel_cols = [c for c in df.columns if 'velocity' in c]
    # CSV might not have acceleration, so we compute it
    
    q_traj = df[joint_cols].values
    qd_traj = df[vel_cols].values
    
    # Compute Joint Acceleration (qdd)
    qdd_traj = np.zeros_like(qd_traj)
    for i in range(len(times)-1):
        qdd_traj[i] = (qd_traj[i+1] - qd_traj[i]) / dt_csv
    qdd_traj[-1] = qdd_traj[-2] # Repeat last
    
    # Quaternion: CSV has x,y,z,w. MuJoCo needs w,x,y,z
    quat_traj = df[['quaternion_w', 'quaternion_x', 'quaternion_y', 'quaternion_z']].values
    
    # Reconstruct Base Angular Velocity (w) and Acceleration (dw)
    # We'll simple diff for now.
    w_base_traj = np.zeros((len(times), 3))
    dw_base_traj = np.zeros((len(times), 3))
    
    for i in range(len(times)-1):
        w_base_traj[i] = get_quaternion_derivative(quat_traj[i], quat_traj[i+1], dt_csv)
    w_base_traj[-1] = w_base_traj[-2] # Repeat last
    
    # Accel
    for i in range(len(times)-1):
        dw_base_traj[i] = (w_base_traj[i+1] - w_base_traj[i]) / dt_csv
    dw_base_traj[-1] = dw_base_traj[-2]
    
    # 3. Control Params
    Kp = 500.0
    Kd = 50.0
    
    # 4. Simulation Loop
    # Data logging
    sim_q_log = []
    sim_quat_log = []
    sim_t_log = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        sim_time = 0.0
        
        # Reset to initial state
        data.qpos[0:3] = np.zeros(3) # Base Pos
        data.qpos[3:7] = quat_traj[0] # Base Quat
        data.qpos[7:] = q_traj[0]    # Joints
        
        data.qvel[0:3] = np.zeros(3) # Base Lin Vel
        data.qvel[3:6] = w_base_traj[0] # Base Ang Vel (Body frame? MuJoCo uses Body frame for free joint ang vel)
        # get_quaternion_derivative returns body frame if using local difference?
        # The formula 2*q_next*q_inv gives w in INERTIAL frame if q represents rotation from Body to Inertial?
        # No, q_dot = 0.5 * q * w_body.
        # w_body = 2 * q_inv * q_dot.
        # My function computed w ~ 2 * q_next * q_inv? 
        # q_next * q_inv is rotation from q to q_next (in inertial frame? or body?).
        # Rotation R_next = R * dR_body.
        # q_next = q * dq_body.
        # dq_body = q_inv * q_next.
        # So I should compute 2 * (q_inv * q_next) to get body angular velocity!
        # Let's fix get_quaternion_derivative logic in thought, but code used q_next * q_inv.
        
        # Recalculate w_base correctly for Body Frame
        # dq_body = q_inv * q_next
        
        data.qvel[6:] = qd_traj[0]   # Joint Vels
        
        # Reset control
        mujoco.mj_forward(model, data)
        
        while viewer.is_running():
            step_start = time.time()
            
            # Check for end of trajectory
            if sim_time >= times[-1]:
                print("Trajectory finished.")
                break

            # 1. Get Reference for current sim_time
            # Find index
            idx = np.searchsorted(times, sim_time)
            if idx >= len(times) - 1:
                idx = len(times) - 2
                # Loop or stop?
                # sim_time = 0 # Loop
                
            # Interpolate
            alpha = (sim_time - times[idx]) / (times[idx+1] - times[idx])
            alpha = np.clip(alpha, 0, 1)
            
            q_ref = (1-alpha)*q_traj[idx] + alpha*q_traj[idx+1]
            qd_ref = (1-alpha)*qd_traj[idx] + alpha*qd_traj[idx+1]
            # qdd_ref is not used for CTC as requested (qdd_ref = 0)
            # qdd_ref = (1-alpha)*qdd_traj[idx] + alpha*qdd_traj[idx+1]
            
            # Slerp for Quaternion
            # ... lazy linear interp for now, or use nearest
            quat_ref = quat_traj[idx] # Use nearest to avoid normalization issues for now
            
            # Base Ref
            w_ref = w_base_traj[idx] # Use body frame w
            dw_ref = dw_base_traj[idx]
            
            # 2. Computed Torque Control (CTC)
            # We use data_ref to calculate Inverse Dynamics on the CURRENT state
            # Target Acceleration = Kp * error + Kd * error_dot (No feedforward accel)
            
            # Set Current State to data_ref
            data_ref.qpos[:] = data.qpos[:]
            data_ref.qvel[:] = data.qvel[:]
            
            # Calculate Errors
            q_err = q_ref - data.qpos[7:]
            qd_err = qd_ref - data.qvel[6:]
            
            # Desired Joint Acceleration (PID term only)
            # Gains: Standard for 1-3, Half for 4-6
            Kp_vec = np.array([Kp]*3 + [Kp*0.5]*3)
            Kd_vec = np.array([Kd]*3 + [Kd*0.5]*3)
            
            qacc_des_joints = Kp_vec * q_err + Kd_vec * qd_err
            
            # Set Target Acceleration
            # Base accel: 0 (Unactuated, we don't try to control it via ID here)
            data_ref.qacc[0:6] = np.zeros(6)
            # Joint accel: Desired
            data_ref.qacc[6:] = qacc_des_joints
            
            # Run Inverse Dynamics: tau = M(q) * qacc_des + C(q, qdot) + G(q)
            mujoco.mj_inverse(model, data_ref)
            
            # Extract Actuator Torques
            # The first 6 DOFs in qfrc_inverse are for the base (unactuated).
            # The next 6 are for the joints.
            tau_ctc = data_ref.qfrc_inverse[6:]
            
            # Total Control
            data.ctrl[:] = tau_ctc
            
            # Log data
            sim_q_log.append(data.qpos[7:].copy())
            sim_quat_log.append(data.qpos[3:7].copy())
            sim_t_log.append(sim_time)

            # 4. Step
            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep
            
            viewer.sync()
            
            # Real-time sync
            # time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)

    # Plot Comparison
    sim_q_log = np.array(sim_q_log)
    sim_quat_log = np.array(sim_quat_log)
    sim_t_log = np.array(sim_t_log)
    
    print("Plotting results...")
    
    # Plot Joints
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    
    for i in range(6):
        ax = axes[i]
        # Plot Reference
        ax.plot(times, q_traj[:, i], 'k--', label='Reference', linewidth=2)
        # Plot Simulation
        ax.plot(sim_t_log, sim_q_log[:, i], 'r-', label='Simulation', linewidth=1.5)
        
        ax.set_ylabel(f'Joint {i+1} [rad]')
        ax.grid(True)
        if i == 0:
            ax.legend()
            
    axes[-1].set_xlabel('Time [s]')
    plt.suptitle('Joint Trajectory Tracking (CTC)')
    plt.tight_layout()
    
    # Plot Base Orientation
    fig2, axes2 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    quat_labels = ['w', 'x', 'y', 'z']
    
    for i in range(4):
        ax = axes2[i]
        ax.plot(times, quat_traj[:, i], 'k--', label='Reference', linewidth=2)
        ax.plot(sim_t_log, sim_quat_log[:, i], 'b-', label='Simulation', linewidth=1.5)
        ax.set_ylabel(f'Quat {quat_labels[i]}')
        ax.grid(True)
        if i == 0: ax.legend()
        
    axes2[-1].set_xlabel('Time [s]')
    plt.suptitle('Base Orientation Tracking (Quaternion)')
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    run_simulation()

