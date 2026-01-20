"""
JAX implementation of SPART (Space Robot Dynamics) functions.
This module provides JAX-compatible versions of SPART dynamics functions
for use with JAX-based optimization and differentiation.

Usage:
    from src.dynamics.urdf2robot_jax import urdf2robot
    from src.dynamics.spart_functions_jax import kinematics, diff_kinematics, etc.
    
    # Load robot model
    robot, _ = urdf2robot('path/to/robot.urdf')
    
    # Use SPART functions
    R0 = jnp.eye(3)
    r0 = jnp.zeros(3)
    qm = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    RJ, RL, rJ, rL, e, g = kinematics(R0, r0, qm, robot)
    Bij, Bi0, P0, pm = diff_kinematics(R0, r0, rL, e, g, robot)
    # ... etc
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from typing import Dict, Tuple, Any

# Import urdf2robot for convenience
try:
    from .urdf2robot_jax import urdf2robot
except ImportError:
    # If relative import fails, try absolute
    try:
        from src.dynamics.urdf2robot_jax import urdf2robot
    except ImportError:
        pass  # urdf2robot is optional

"""
t0 : [6x1] Base-link twist projected in the inertial coordinate system (CCS). The first three elements represent the angular velocity, and the last three represent the linear velocity.
tL : [6xn] Twists of the links projected in the inertial CCS.
P0 : [6x6] Base-link twist-propagation matrix, which transforms base-link velocities into generalized velocities.
pm : [6xn] Manipulator twist-propagation matrix, defining the effect of joint velocities on the system's motion.
Bi0 : [6x6xn] Twist-propagation matrix describing the relationship between base-link motion and link motion.
Bij : [6x6xnxn] Twist-propagation matrix defining the motion coupling between different links.
u0 : [6x1] Base-link velocities [ω, r_dot]. Angular velocity is expressed in a body-fixed CCS, while linear velocity is in the inertial CCS.
um : [nx1] Joint velocities, representing the motion of each joint in the system.
u0dot : [6x1] Base-link accelerations [ω_dot, r_double_dot]. Angular acceleration is expressed in a body-fixed CCS, while linear acceleration is in the inertial CCS.
umdot : [nx1] Joint accelerations.
robot : [struct] Robot model containing parameters such as number of links, joints, mass, inertia, connectivity, and kinematic information.
t0dot : [6x1] Time derivative of the base-link twist, indicating the rate of change of twist.
tLdot : [6xn] Time derivative of the link twists, representing their change over time.
R0 : [3x3] Rotation matrix transforming base-link coordinates into the inertial CCS.
r0 : [3x1] Position of the base-link center-of-mass relative to the inertial frame.
RJ : [3x3xn] Rotation matrices for each joint with respect to the inertial CCS.
RL : [3x3xn] Rotation matrices for each link with respect to the inertial CCS.
rJ : [3xn] Positions of the joints projected in the inertial CCS.
rL : [3xn] Positions of the links projected in the inertial CCS.
e : [3xn] Joint rotation/sliding axes projected in the inertial CCS.
g : [3xn] Vector from the ith joint's origin to the ith link's origin, projected in the inertial CCS.
Bij : [6x6xn] Twist-propagation matrix for manipulator links.
Bi0 : [6x6xn] Twist-propagation matrix between the base and the links.
I0 : [3x3] Inertia matrix of the base-link, projected in the inertial CCS.
Im : [3x3xn] Inertia matrices of the links, projected in the inertial CCS.
M0_tilde : [6x6] Composite mass matrix of the base-link.
Mm_tilde : [6x6xn] Composite mass matrices of the manipulator links.
H0 : [6x6] Base-link inertia matrix.
H0m : [6xn_q] Base-link to manipulator coupling inertia matrix.
Hm : [n_qxn_q] Manipulator inertia matrix.
"""


def skew_symmetric(v):
    """Convert 3D vector to skew-symmetric matrix [v]_x."""
    v = jnp.asarray(v).flatten()
    assert len(v) == 3, 'vector length error'
    
    return jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def euler_dcm(e, alpha):
    """Euler axis-angle to DCM (Direction Cosine Matrix)."""
    e = jnp.asarray(e)
    alpha = jnp.asarray(alpha)
    q = jnp.hstack((e * jnp.sin(alpha / 2), jnp.cos(alpha / 2)))
    return quat_dcm(q)


def quat_dcm(q):
    """Convert quaternion [q1, q2, q3, q0] = [x, y, z, w] to rotation matrix."""
    q = jnp.asarray(q).flatten()
    assert len(q) == 4, 'quaternion length error'
    q1, q2, q3, q0 = q
    
    return jnp.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)]
    ])


def quat_dot(q, w):
    """
    Quaternion derivative: dq/dt = 0.5 * q * [0, wx, wy, wz]
    where q = [q1, q2, q3, q0] = [x, y, z, w] and w = [wx, wy, wz]
    """
    q = jnp.asarray(q).flatten()
    w = jnp.asarray(w).flatten()
    assert len(q) == 4 and len(w) == 3, 'quaternion or angular velocity length error'
    q1, q2, q3, q0 = q  # [x, y, z, w]
    w1, w2, w3 = w      # [wx, wy, wz]
    
    return jnp.array([
        0.5 * (q0 * w1 + q2 * w3 - q3 * w2),  # q_dot[0] = x_dot
        0.5 * (q0 * w2 + q3 * w1 - q1 * w3),  # q_dot[1] = y_dot
        0.5 * (q0 * w3 + q1 * w2 - q2 * w1),  # q_dot[2] = z_dot
        0.5 * (-q1 * w1 - q2 * w2 - q3 * w3)  # q_dot[3] = w_dot
    ])


def kinematics(R0, r0, qm, robot):
    """
    Forward kinematics computation.
    
    Args:
        R0: [3, 3] base rotation matrix
        r0: [3] base position
        qm: [n_q] joint angles
        robot: robot model dictionary
    
    Returns:
        RJ: [3, 3, n] joint rotation matrices
        RL: [3, 3, n] link rotation matrices
        rJ: [3, n] joint positions
        rL: [3, n] link positions
        e: [3, n] joint axes
        g: [3, n] link vectors
    """
    n = robot['n_links_joints']
    
    # T0 transformation matrix
    T0 = jnp.block([
        [R0, r0.reshape(3, 1)],
        [jnp.array([0, 0, 0, 1])]
    ])
    
    TJ_list = []
    TL_list = []
    
    for i in range(n):
        joint = robot['joints'][i]
        joint_T = jnp.asarray(joint['T'])
        
        if joint['parent_link'] == 0:
            TJ_val = T0 @ joint_T
        else:
            TJ_val = TL_list[joint['parent_link'] - 1] @ joint_T
        
        TJ_list.append(TJ_val)
        
        # Joint transformation based on joint type
        T_qm = jnp.eye(4)
        
        if joint['type'] == 1:  # Revolute
            axis = jnp.asarray(joint['axis'])
            R_qm = euler_dcm(axis, qm[joint['q_id'] - 1])
            T_qm = jnp.block([
                [R_qm, jnp.zeros((3, 1))],
                [jnp.array([[0, 0, 0, 1]])]
            ])
        elif joint['type'] == 2:  # Prismatic
            axis = jnp.asarray(joint['axis'])
            p_qm = (axis.flatten() * qm[joint['q_id'] - 1]).reshape(3, 1)
            T_qm = jnp.block([
                [jnp.eye(3), p_qm],
                [jnp.array([[0, 0, 0, 1]])]
            ])
        
        link = robot['links'][joint['child_link'] - 1]
        link_T = jnp.asarray(link['T'])
        
        TL_val = TJ_val @ T_qm @ link_T
        TL_list.append(TL_val)
    
    # Stack into arrays
    TJ = jnp.stack(TJ_list, axis=2)
    TL = jnp.stack(TL_list, axis=2)
    
    RJ = TJ[:3, :3, :]
    RL = TL[:3, :3, :]
    rJ = TJ[:3, 3, :]
    rL = TL[:3, 3, :]
    
    # Compute joint axes and link vectors
    e_list = []
    g_list = []
    
    for i in range(n):
        axis = jnp.asarray(robot['joints'][i]['axis'])
        e_val = RJ[:, :, i] @ axis
        e_list.append(e_val)
        
        pj_idx = robot['links'][i]['parent_joint'] - 1
        g_val = rL[:, i] - rJ[:, pj_idx]
        g_list.append(g_val)
    
    e = jnp.stack(e_list, axis=1)
    g = jnp.stack(g_list, axis=1)
    
    return RJ, RL, rJ, rL, e, g


def diff_kinematics(R0, r0, rL, e, g, robot):
    """
    Differential kinematics computation.
    
    Args:
        R0: [3, 3] base rotation matrix
        r0: [3] base position
        rL: [3, n] link positions
        e: [3, n] joint axes
        g: [3, n] link vectors
        robot: robot model dictionary
    
    Returns:
        Bij: [6, 6, n, n] link-to-link twist propagation matrices
        Bi0: [6, 6, n] base-to-link twist propagation matrices
        P0: [6, 6] base twist propagation matrix
        pm: [6, n] manipulator twist propagation vectors
    """
    n = robot['n_links_joints']
    
    # P0 matrix
    zeros_33 = jnp.zeros((3, 3))
    I3 = jnp.eye(3)
    P0 = jnp.block([
        [R0, zeros_33],
        [zeros_33, I3]
    ])
    
    # Bij calculation - vectorized
    rL_j = rL[:, :, jnp.newaxis]  # [3, n, 1]
    rL_i = rL[:, jnp.newaxis, :]   # [3, 1, n]
    rL_diff = rL_i - rL_j           # [3, n, n]
    
    # Skew symmetric matrices for all pairs
    rx = rL_diff[0]  # [n, n]
    ry = rL_diff[1]
    rz = rL_diff[2]
    
    zeros_nn = jnp.zeros((n, n))
    
    row0 = jnp.stack([zeros_nn, -rz, ry], axis=0)  # [3, n, n]
    row1 = jnp.stack([rz, zeros_nn, -rx], axis=0)
    row2 = jnp.stack([-ry, rx, zeros_nn], axis=0)
    
    skew_all = jnp.stack([row0, row1, row2], axis=0)  # [3, 3, n, n]
    
    # Bij block matrix [[I, 0], [skew, I]]
    I3_expand = jnp.expand_dims(jnp.expand_dims(I3, -1), -1)  # [3, 3, 1, 1]
    I3_expand = jnp.broadcast_to(I3_expand, (3, 3, n, n))
    zeros_expand = jnp.zeros((3, 3, n, n))
    
    row1 = jnp.concatenate([I3_expand, zeros_expand], axis=1)  # [3, 6, n, n]
    row2 = jnp.concatenate([skew_all, I3_expand], axis=1)     # [3, 6, n, n]
    Bij = jnp.concatenate([row1, row2], axis=0)               # [6, 6, n, n]
    
    # Apply branch mask
    branch_mask = jnp.asarray(robot['con']['branch'])
    Bij = Bij * branch_mask[jnp.newaxis, jnp.newaxis, :, :]
    
    # Bi0 calculation
    r0_expand = r0.reshape(3, 1)
    r0_diff = r0_expand - rL  # [3, n]
    
    rx0 = r0_diff[0]  # [n]
    ry0 = r0_diff[1]
    rz0 = r0_diff[2]
    
    zeros_n = jnp.zeros(n)
    
    row0_Bi0 = jnp.stack([zeros_n, -rz0, ry0], axis=0)  # [3, n]
    row1_Bi0 = jnp.stack([rz0, zeros_n, -rx0], axis=0)
    row2_Bi0 = jnp.stack([-ry0, rx0, zeros_n], axis=0)
    
    skew_Bi0 = jnp.stack([row0_Bi0, row1_Bi0, row2_Bi0], axis=0)  # [3, 3, n]
    
    I3_n = jnp.expand_dims(I3, -1)  # [3, 3, 1]
    I3_n = jnp.broadcast_to(I3_n, (3, 3, n))
    zeros_n_33 = jnp.zeros((3, 3, n))
    
    Bi0_row1 = jnp.concatenate([I3_n, zeros_n_33], axis=1)  # [3, 6, n]
    Bi0_row2 = jnp.concatenate([skew_Bi0, I3_n], axis=1)    # [3, 6, n]
    Bi0 = jnp.concatenate([Bi0_row1, Bi0_row2], axis=0)    # [6, 6, n]
    
    # pm calculation
    cross_eg = jnp.cross(e, g, axis=0)  # [3, n]
    
    pm_list = []
    for i in range(n):
        jt = robot['joints'][i]['type']
        if jt == 1:  # Revolute
            pm_i = jnp.concatenate([e[:, i], cross_eg[:, i]], axis=0)
        elif jt == 2:  # Prismatic
            pm_i = jnp.concatenate([jnp.zeros(3), e[:, i]], axis=0)
        else:
            pm_i = jnp.zeros(6)
        pm_list.append(pm_i)
    
    pm = jnp.stack(pm_list, axis=1)  # [6, n]
    
    return Bij, Bi0, P0, pm


def inertia_projection(R0, RL, robot):
    """
    Project inertia matrices to inertial frame.
    
    Args:
        R0: [3, 3] base rotation matrix
        RL: [3, 3, n] link rotation matrices
        robot: robot model dictionary
    
    Returns:
        I0: [3, 3] base inertia in inertial frame
        Im: [3, 3, n] link inertias in inertial frame
    """
    base_inertia = jnp.asarray(robot['base_link']['inertia'])
    I0 = R0 @ base_inertia @ R0.T
    
    n = robot['n_links_joints']
    Im_list = []
    for i in range(n):
        link_inertia = jnp.asarray(robot['links'][i]['inertia'])
        val = RL[:, :, i] @ link_inertia @ RL[:, :, i].T
        Im_list.append(val)
    
    Im = jnp.stack(Im_list, axis=2)
    return I0, Im


def mass_composite_body(I0, Im, Bij, Bi0, robot):
    """
    Compute composite body mass matrices.
    
    Args:
        I0: [3, 3] base inertia
        Im: [3, 3, n] link inertias
        Bij: [6, 6, n, n] link-to-link propagation matrices
        Bi0: [6, 6, n] base-to-link propagation matrices
        robot: robot model dictionary
    
    Returns:
        M0_tilde: [6, 6] base composite mass matrix
        Mm_tilde: [6, 6, n] link composite mass matrices
    """
    n = robot['n_links_joints']
    
    # Compute Mm_tilde in reverse order
    # Initialize with zeros instead of None for JAX compatibility
    Mm_tilde_list = [jnp.zeros((6, 6)) for _ in range(n)]
    
    for i in reversed(range(n)):
        Im_i = Im[:, :, i]
        mass_i = jnp.asarray(robot['links'][i]['mass'])
        zeros = jnp.zeros((3, 3))
        I3 = jnp.eye(3)
        
        val = jnp.block([
            [Im_i, zeros],
            [zeros, mass_i * I3]
        ])
        
        # Add contributions from children
        # Use JAX-compatible indexing - iterate through all possible children
        # Note: we process in reverse order, so j > i means Mm_tilde_list[j] is already computed
        child_mask = jnp.asarray(robot['con']['child'][:, i])
        for j in range(n):
            # Only process if j > i (already computed) and child_mask[j] == 1
            # Use jnp.where for JAX compatibility
            is_child = (child_mask[j] == 1) & (j > i)
            # Compute term only if j > i (already computed)
            term = Bij[:, :, j, i].T @ Mm_tilde_list[j] @ Bij[:, :, j, i]
            val = val + jnp.where(is_child, term, jnp.zeros_like(term))
        
        Mm_tilde_list[i] = val
    
    Mm_tilde = jnp.stack(Mm_tilde_list, axis=2)
    
    # Compute M0_tilde
    mass_base = jnp.asarray(robot['base_link']['mass'])
    zeros = jnp.zeros((3, 3))
    I3 = jnp.eye(3)
    
    M0_tilde = jnp.block([
        [I0, zeros],
        [zeros, mass_base * I3]
    ])
    
    child_base_mask = jnp.asarray(robot['con']['child_base'])
    n = robot['n_links_joints']
    for j in range(n):
        # Use jnp.where for JAX compatibility
        term = Bi0[:, :, j].T @ Mm_tilde_list[j] @ Bi0[:, :, j]
        M0_tilde = M0_tilde + jnp.where(child_base_mask[j] == 1, term, jnp.zeros_like(term))
    
    return M0_tilde, Mm_tilde


def generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot):
    """
    Compute generalized inertia matrices.
    
    Args:
        M0_tilde: [6, 6] base composite mass matrix
        Mm_tilde: [6, 6, n] link composite mass matrices
        Bij: [6, 6, n, n] link-to-link propagation matrices
        Bi0: [6, 6, n] base-to-link propagation matrices
        P0: [6, 6] base propagation matrix
        pm: [6, n] manipulator propagation vectors
        robot: robot model dictionary
    
    Returns:
        H0: [6, 6] base inertia matrix
        H0m: [6, n_q] base-manipulator coupling matrix
        Hm: [n_q, n_q] manipulator inertia matrix
    """
    n_q = robot['n_q']
    n = robot['n_links_joints']
    
    H0 = P0.T @ M0_tilde @ P0
    
    # Collect active joint indices
    active_indices = []
    q_ids = []
    for i in range(n):
        if robot['joints'][i]['type'] != 0:
            active_indices.append(i)
            q_ids.append(robot['joints'][i]['q_id'] - 1)
    
    num_active = len(active_indices)
    
    if num_active == 0:
        return H0, jnp.zeros((6, n_q)), jnp.zeros((n_q, n_q))
    
    active_idx_array = jnp.array(active_indices)
    
    # Gather active components
    pm_active = pm[:, active_idx_array]  # [6, num_active]
    Mm_tilde_active = Mm_tilde[:, :, active_idx_array]  # [6, 6, num_active]
    
    # Create grid for Bij
    grid_i, grid_j = jnp.meshgrid(active_idx_array, active_idx_array, indexing='ij')
    Bij_active = Bij[:, :, grid_i, grid_j]  # [6, 6, num_active, num_active]
    
    # Compute Hm using einsum
    Mm_pm = jnp.einsum('mnk,nk->mk', Mm_tilde_active, pm_active)  # [6, num_active]
    Bij_pm = jnp.einsum('mnij,nj->mij', Bij_active, pm_active)     # [6, num_active, num_active]
    Hm_dense = jnp.einsum('mi,mij->ij', Mm_pm, Bij_pm)             # [num_active, num_active]
    
    # Symmetrize
    diagonal = jnp.diag(Hm_dense)
    Hm_dense_sym = Hm_dense + Hm_dense.T - jnp.diag(diagonal)
    
    # Check if q_ids are contiguous
    is_contiguous = (num_active == n_q)
    if is_contiguous:
        for i in range(num_active):
            if q_ids[i] != i:
                is_contiguous = False
                break
    
    if is_contiguous:
        Hm = Hm_dense_sym
    else:
        # Scatter to correct positions
        Hm = jnp.zeros((n_q, n_q))
        for i_idx, qi in enumerate(q_ids):
            for j_idx, qj in enumerate(q_ids):
                Hm = Hm.at[qi, qj].set(Hm_dense_sym[i_idx, j_idx])
    
    # Compute H0m
    Bi0_active = Bi0[:, :, active_idx_array]  # [6, 6, num_active]
    term1 = jnp.einsum('mnk,mk->nk', Bi0_active, Mm_pm)  # [6, num_active]
    H0m_dense = P0.T @ term1  # [6, num_active]
    
    if is_contiguous:
        H0m = H0m_dense
    else:
        H0m = jnp.zeros((6, n_q))
        for i_idx, qi in enumerate(q_ids):
            H0m = H0m.at[:, qi].set(H0m_dense[:, i_idx])
    
    return H0, H0m, Hm

