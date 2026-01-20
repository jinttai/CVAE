"""
Comparison test between JAX and PyTorch urdf2robot implementations.
This script tests numerical equivalence between the two implementations.
"""

import torch
import jax.numpy as jnp
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from src.dynamics.urdf2robot_torch import urdf2robot as urdf2robot_torch
from src.dynamics.urdf2robot_jax import urdf2robot as urdf2robot_jax


def compare_arrays(name, arr1, arr2, rtol=1e-5, atol=1e-6):
    """
    Compare two arrays and print results.
    
    Args:
        name: Name of the comparison
        arr1: First array (PyTorch tensor or numpy array)
        arr2: Second array (JAX array or numpy array)
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    # Convert to numpy
    if isinstance(arr1, torch.Tensor):
        arr1_np = arr1.detach().cpu().numpy()
    else:
        arr1_np = np.asarray(arr1)
    
    if isinstance(arr2, jnp.ndarray):
        arr2_np = np.asarray(arr2)
    else:
        arr2_np = np.asarray(arr2)
    
    # Check shapes
    if arr1_np.shape != arr2_np.shape:
        print(f"[FAIL] {name}: Shape mismatch! {arr1_np.shape} vs {arr2_np.shape}")
        return False
    
    # Compute errors
    abs_error = np.abs(arr1_np - arr2_np)
    rel_error = np.abs(arr1_np - arr2_np) / (np.abs(arr1_np) + 1e-10)
    
    max_abs_error = np.max(abs_error)
    max_rel_error = np.max(rel_error)
    mean_abs_error = np.mean(abs_error)
    mean_rel_error = np.mean(rel_error)
    
    # Check if within tolerance
    is_close = np.allclose(arr1_np, arr2_np, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"[PASS] {name}: PASSED")
        print(f"   Max abs error: {max_abs_error:.2e}, Mean abs error: {mean_abs_error:.2e}")
        print(f"   Max rel error: {max_rel_error:.2e}, Mean rel error: {mean_rel_error:.2e}")
    else:
        print(f"[FAIL] {name}: FAILED")
        print(f"   Max abs error: {max_abs_error:.2e}, Mean abs error: {mean_abs_error:.2e}")
        print(f"   Max rel error: {max_rel_error:.2e}, Mean rel error: {mean_rel_error:.2e}")
        print(f"   Shape: {arr1_np.shape}")
        if arr1_np.size < 20:
            print(f"   PyTorch: {arr1_np}")
            print(f"   JAX:     {arr2_np}")
    
    return is_close


def compare_scalars(name, val1, val2, rtol=1e-5, atol=1e-6):
    """Compare two scalar values."""
    if isinstance(val1, torch.Tensor):
        val1 = float(val1.detach().cpu().numpy())
    if isinstance(val2, jnp.ndarray):
        val2 = float(np.asarray(val2))
    
    abs_error = abs(val1 - val2)
    rel_error = abs_error / (abs(val1) + 1e-10)
    
    is_close = abs_error < atol or rel_error < rtol
    
    if is_close:
        print(f"[PASS] {name}: PASSED (abs error: {abs_error:.2e}, rel error: {rel_error:.2e})")
    else:
        print(f"[FAIL] {name}: FAILED (abs error: {abs_error:.2e}, rel error: {rel_error:.2e})")
        print(f"   PyTorch: {val1}, JAX: {val2}")
    
    return is_close


def test_urdf2robot():
    """Test urdf2robot function."""
    print("="*60)
    print("URDF2Robot JAX vs PyTorch Comparison Test")
    print("="*60)
    
    # Load robot model
    ROOT_DIR = os.path.dirname(__file__)
    urdf_path = os.path.join(ROOT_DIR, "assets/SC_ur10e.urdf")
    
    if not os.path.exists(urdf_path):
        urdf_path = os.path.join(ROOT_DIR, "src/dynamics/assets/SC_ur10e.urdf")
    
    if not os.path.exists(urdf_path):
        print(f"[ERROR] Could not find URDF file at {urdf_path}")
        return
    
    print(f"\nLoading robot model from: {urdf_path}")
    
    # PyTorch
    device = 'cpu'
    robot_torch, keys_torch = urdf2robot_torch(urdf_path, verbose_flag=False, device=device)
    
    # JAX
    robot_jax, keys_jax = urdf2robot_jax(urdf_path, verbose_flag=False)
    
    print(f"Robot: {robot_torch['n_links_joints']} links/joints, {robot_torch['n_q']} DOF")
    
    all_passed = True
    
    # Test basic properties
    print("\n" + "="*60)
    print("Testing basic properties")
    print("="*60)
    all_passed &= compare_scalars("n_q", robot_torch['n_q'], robot_jax['n_q'])
    all_passed &= compare_scalars("n_links_joints", robot_torch['n_links_joints'], robot_jax['n_links_joints'])
    all_passed &= compare_scalars("base_link mass", robot_torch['base_link']['mass'], robot_jax['base_link']['mass'])
    
    # Test base_link inertia
    print("\n" + "="*60)
    print("Testing base_link inertia")
    print("="*60)
    all_passed &= compare_arrays("base_link inertia", robot_torch['base_link']['inertia'], robot_jax['base_link']['inertia'])
    
    # Test links
    print("\n" + "="*60)
    print("Testing links")
    print("="*60)
    n = robot_torch['n_links_joints']
    for i in range(n):
        link_torch = robot_torch['links'][i]
        link_jax = robot_jax['links'][i]
        
        all_passed &= compare_scalars(f"link[{i}] id", link_torch['id'], link_jax['id'])
        all_passed &= compare_scalars(f"link[{i}] mass", link_torch['mass'], link_jax['mass'])
        all_passed &= compare_arrays(f"link[{i}] T", link_torch['T'], link_jax['T'])
        all_passed &= compare_arrays(f"link[{i}] inertia", link_torch['inertia'], link_jax['inertia'])
    
    # Test joints
    print("\n" + "="*60)
    print("Testing joints")
    print("="*60)
    for i in range(n):
        joint_torch = robot_torch['joints'][i]
        joint_jax = robot_jax['joints'][i]
        
        all_passed &= compare_scalars(f"joint[{i}] id", joint_torch['id'], joint_jax['id'])
        all_passed &= compare_scalars(f"joint[{i}] type", joint_torch['type'], joint_jax['type'])
        all_passed &= compare_scalars(f"joint[{i}] q_id", joint_torch['q_id'], joint_jax['q_id'])
        all_passed &= compare_scalars(f"joint[{i}] parent_link", joint_torch['parent_link'], joint_jax['parent_link'])
        all_passed &= compare_scalars(f"joint[{i}] child_link", joint_torch['child_link'], joint_jax['child_link'])
        all_passed &= compare_arrays(f"joint[{i}] T", joint_torch['T'], joint_jax['T'])
        all_passed &= compare_arrays(f"joint[{i}] axis", joint_torch['axis'], joint_jax['axis'])
        all_passed &= compare_arrays(f"joint[{i}] limit", joint_torch['limit'], joint_jax['limit'])
    
    # Test connectivity maps
    print("\n" + "="*60)
    print("Testing connectivity maps")
    print("="*60)
    all_passed &= compare_arrays("con branch", robot_torch['con']['branch'], robot_jax['con']['branch'])
    all_passed &= compare_arrays("con child", robot_torch['con']['child'], robot_jax['con']['child'])
    all_passed &= compare_arrays("con child_base", robot_torch['con']['child_base'], robot_jax['con']['child_base'])
    
    # Test joint limits
    print("\n" + "="*60)
    print("Testing joint limits")
    print("="*60)
    all_passed &= compare_arrays("joint_limits", robot_torch['joint_limits'], robot_jax['joint_limits'])
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if all_passed:
        print("[SUCCESS] All tests PASSED!")
    else:
        print("[FAILURE] Some tests FAILED!")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    test_urdf2robot()

