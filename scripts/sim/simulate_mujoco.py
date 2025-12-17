"""
MuJoCo Simulation for UR10e Robot using trajectory from CSV file.
(Video Rendering Fixed Version)

This script loads the URDF model and simulates the robot following
the joint trajectory specified in the CSV file.
"""

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import os
import sys
import time

# Add root directory helper
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. Video saving will be disabled.")


def euler_to_quaternion(yaw, pitch, roll):
    """Convert Euler angles (ZYX convention) to quaternion (w, x, y, z)."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def load_trajectory(csv_path, orientation_csv_path=None):
    """Load trajectory and interpolate if necessary."""
    df = pd.read_csv(csv_path)
    
    # Extract time and joint angles
    times = df['t'].values
    joint_angles = df[['J1', 'J2', 'J3', 'J4', 'J5', 'J6']].values
    
    print(f"Loaded trajectory: {len(times)} steps, duration: {times[-1]:.2f}s")
    
    # Load base orientation if provided
    base_orientations = None
    if orientation_csv_path and os.path.exists(orientation_csv_path):
        print(f"Loading base orientation from: {orientation_csv_path}")
        orient_df = pd.read_csv(orientation_csv_path)
        
        # Check if time columns match
        if len(orient_df) != len(times):
            print(f"Warning: Interpolating orientation to match trajectory times...")
            yaw_interp = np.interp(times, orient_df['t'].values, orient_df['yaw'].values)
            pitch_interp = np.interp(times, orient_df['t'].values, orient_df['pitch'].values)
            roll_interp = np.interp(times, orient_df['t'].values, orient_df['roll'].values)
        else:
            yaw_interp = orient_df['yaw'].values
            pitch_interp = orient_df['pitch'].values
            roll_interp = orient_df['roll'].values
        
        # Convert Euler angles to quaternions
        base_orientations = np.zeros((len(times), 4))
        for i in range(len(times)):
            base_orientations[i] = euler_to_quaternion(yaw_interp[i], pitch_interp[i], roll_interp[i])
            
    return times, joint_angles, base_orientations


def find_joint_ids(model, expected_joint_names):
    """Find joint IDs for expected joint names."""
    joint_info = []
    for joint_name in expected_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            qpos_addr = model.jnt_qposadr[joint_id]
            joint_info.append((joint_id, qpos_addr))
        else:
            print(f"Warning: Joint '{joint_name}' not found in model")
    return joint_info


def simulate_trajectory(model_path, trajectory_csv_path, playback_speed=1.0, save_video=False, orientation_csv_path=None):
    """Simulate robot following trajectory and save video."""
    
    # 1. Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading MuJoCo model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 2. Setup Joints
    expected_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    joint_info = find_joint_ids(model, expected_joint_names)
    
    if len(joint_info) < 6:
        print("Error: Could not find all 6 joints. Check URDF.")
        return

    # 3. Load Trajectory Data
    times, joint_angles, base_orientations = load_trajectory(trajectory_csv_path, orientation_csv_path)
    
    # 4. Interpolation Setup
    sim_dt = model.opt.timestep
    total_time = times[-1]
    n_sim_steps = int(total_time / sim_dt)
    sim_times = np.linspace(0, total_time, n_sim_steps)
    
    interpolated_angles = np.zeros((n_sim_steps, 6))
    for i in range(6):
        interpolated_angles[:, i] = np.interp(sim_times, times, joint_angles[:, i])
    
    # Add offset to shoulder_lift_joint as per original code
    interpolated_angles[:, 1] -= np.pi / 2.0
    
    interpolated_orientations = None
    if base_orientations is not None:
        interpolated_orientations = np.zeros((n_sim_steps, 4))
        for i in range(4):
            interpolated_orientations[:, i] = np.interp(sim_times, times, base_orientations[:, i])
        # Normalize quaternions
        norms = np.linalg.norm(interpolated_orientations, axis=1, keepdims=True)
        interpolated_orientations /= norms
    
    # 5. Base Joint Setup (Free Joint)
    base_free_joint_id = None
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    
    # Find free joint for base
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            if model.jnt_bodyid[i] == base_body_id:
                base_free_joint_id = i
                break
    
    if base_free_joint_id is None:
        # Fallback search
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                base_free_joint_id = i
                break

    # Initialize Base
    if base_free_joint_id is not None:
        qpos_addr = model.jnt_qposadr[base_free_joint_id]
        data.qpos[qpos_addr:qpos_addr+3] = [0.0, 0.0, 0.0]
        if interpolated_orientations is not None:
            data.qpos[qpos_addr+3:qpos_addr+7] = interpolated_orientations[0]
        else:
            data.qpos[qpos_addr+3:qpos_addr+7] = [1.0, 0.0, 0.0, 0.0]
    
    # Initialize Arms
    for i, (_, qpos_addr) in enumerate(joint_info):
        data.qpos[qpos_addr] = interpolated_angles[0, i]
        
    mujoco.mj_forward(model, data)

    # 6. Video Recording Setup (Improved with mujoco.Renderer)
    renderer = None
    video_frames = []
    video_camera = mujoco.MjvCamera() # Camera object for settings
    video_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    video_camera.distance = 8.0
    video_camera.azimuth = 45.0
    video_camera.elevation = -20.0
    video_camera.lookat[:] = [0.0, 0.0, 0.5]
    
    if save_video and HAS_IMAGEIO:
         try:
             # Use high-level Renderer class (Standard in modern MuJoCo)
             # Note: XML should have <visual><global offwidth="1920" offheight="1080"/></visual>
             renderer = mujoco.Renderer(model, height=1080, width=1920)
             print("Video recording initialized using mujoco.Renderer.")
         except Exception as e:
             print(f"Failed to initialize renderer: {e}")
             print("Trying with lower resolution...")
             try:
                 # Fallback to lower resolution
                 renderer = mujoco.Renderer(model, height=720, width=1280)
                 print("Video recording initialized with 1280x720 resolution.")
             except Exception as e2:
                 print(f"Failed with lower resolution too: {e2}")
                 save_video = False
                 renderer = None

    print("\nStarting simulation...")
    
    # 7. Simulation Loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        step_idx = 0
        
        while viewer.is_running() and step_idx < n_sim_steps:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 8.0
            viewer.cam.azimuth = 45.0
            viewer.cam.elevation = -20.0
            viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
            loop_start = time.time()
            
            # --- Update Physics State ---
            # Update Base Orientation
            if base_free_joint_id is not None and interpolated_orientations is not None:
                qpos_addr = model.jnt_qposadr[base_free_joint_id]
                data.qpos[qpos_addr+3:qpos_addr+7] = interpolated_orientations[step_idx]

            # Update Joint Angles
            for i, (_, qpos_addr) in enumerate(joint_info):
                data.qpos[qpos_addr] = interpolated_angles[step_idx, i]
            
            # Step Physics
            mujoco.mj_step(model, data)
            
            # --- Update Live Viewer ---
            viewer.sync()
            
            # --- Video Capture ---
            if save_video and renderer is not None:
                # Update camera lookat to follow base
                if base_free_joint_id is not None:
                    qpos_addr = model.jnt_qposadr[base_free_joint_id]
                    base_pos = data.qpos[qpos_addr:qpos_addr+3]
                    video_camera.lookat[:] = base_pos + [0.0, 0.0, 0.5]
                
                try:
                    # Update renderer scene and capture
                    renderer.update_scene(data, camera=video_camera)
                    pixels = renderer.render()
                    video_frames.append(pixels.copy())
                except Exception as e:
                    if step_idx == 0:
                        print(f"Frame capture failed: {e}")
            
            # --- Timing Control ---
            step_idx += 1
            
            # Simple real-time sync logic
            elapsed = time.time() - loop_start
            target_dt = sim_dt / playback_speed
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    # 8. Save Video
    print("\nSimulation finished.")
    if save_video and HAS_IMAGEIO and len(video_frames) > 0:
        video_path = "simulation_video.mp4"
        fps = int(1.0 / sim_dt * playback_speed)
        print(f"Saving {len(video_frames)} frames to {video_path} (FPS: {fps})...")
        try:
            imageio.mimsave(video_path, video_frames, fps=fps, codec='libx264', quality=8)
            print("Video saved successfully!")
        except Exception as e:
            print(f"Error saving video file: {e}")
    else:
        if save_video:
            print("Warning: No frames captured.")


def main():
    # Default paths
    model_path = os.path.join(ROOT_DIR, "assets/spacerobot_cjt.xml")
    trajectory_csv_path = os.path.join(ROOT_DIR, "outputs/results/opt_nn_lbfgs/q_traj.csv")
    orientation_csv_path = os.path.join(ROOT_DIR, "outputs/results/opt_nn_lbfgs/body_orientation.csv")
    
    # CLI Arguments
    if len(sys.argv) > 1: trajectory_csv_path = sys.argv[1]
    if len(sys.argv) > 2: model_path = sys.argv[2]
    if len(sys.argv) > 3: orientation_csv_path = sys.argv[3]
    
    try:
        simulate_trajectory(
            model_path=model_path,
            trajectory_csv_path=trajectory_csv_path,
            orientation_csv_path=orientation_csv_path,
            save_video=True
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()