"""
Trajectory utilities for saving DDP/iLQR trajectories.
Includes CSV export functionality.
"""

import os
import numpy as np
import pandas as pd
import torch


def save_trajectory_csv(states, controls, dt, save_path, method_name="ilqr"):
    """
    Save trajectory data to CSV file.
    
    Args:
        states: [T+1, 16] tensor or numpy array [q (6), qd (6), quaternion (4)]
        controls: [T, 6] tensor or numpy array [qdd (6)]
        dt: time step (seconds)
        save_path: path to save CSV file
        method_name: method name for metadata (e.g., "ilqr", "ddp")
    """
    # Convert to numpy if tensor
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    if isinstance(controls, torch.Tensor):
        controls = controls.detach().cpu().numpy()
    
    T = controls.shape[0]
    time = np.arange(T + 1) * dt
    
    # Prepare data dictionary
    data = {'time': time}
    
    # Joint angles (states[:, :6])
    for i in range(6):
        data[f'joint_{i+1}_angle'] = states[:, i]
    
    # Joint velocities (states[:, 6:12])
    for i in range(6):
        data[f'joint_{i+1}_velocity'] = states[:, 6+i]
    
    # Joint accelerations (controls)
    # Pad controls to match states length (last control value repeated)
    controls_padded = np.vstack([controls, controls[-1:]])
    for i in range(6):
        data[f'joint_{i+1}_acceleration'] = controls_padded[:, i]
    
    # Quaternion (states[:, 12:])
    data['quaternion_x'] = states[:, 12]
    data['quaternion_y'] = states[:, 13]
    data['quaternion_z'] = states[:, 14]
    data['quaternion_w'] = states[:, 15]
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, float_format='%.8f')
    print(f"Trajectory CSV saved to: {save_path}")
    print(f"  Shape: {df.shape}, Columns: {list(df.columns)}")
