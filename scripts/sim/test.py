import sys
import os
import numpy as np
import mujoco

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)
from src.dynamics.urdf2robot import urdf2robot
import src.dynamics.spart_functions as ft

def dcm_to_quat(R):
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    """
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def transform_to_pos_quat(T):
    """Convert 4x4 homogenous transform T to (pos, quat).
    quat = [w, x, y, z] matching MuJoCo convention."""
    pos = T[0:3, 3]
    R = T[0:3, 0:3]
    
    # Rotation to quaternion
    quat = dcm_to_quat(R)
    return pos, quat

def test_kinematics():
    print("\n========================================")
    print("STEP 3: Verify Kinematics (Link Lengths, Joint Axes)")
    print("========================================")

    # 1. Load XML
    xml_path = os.path.join(ROOT_DIR, 'assets/spacerobot_cjt.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)

    # 2. Load URDF
    urdf_path = os.path.join(ROOT_DIR, 'assets/SC_ur10e.urdf')
    try:
        robot, _ = urdf2robot(urdf_path)
    except:
        urdf_path = os.path.join(ROOT_DIR, 'src/dynamics/assets/SC_ur10e.urdf')
        robot, _ = urdf2robot(urdf_path)

    print(f"{'Joint Name':<25} | {'Param':<10} | {'MuJoCo':<25} | {'Spart/URDF':<25} | {'Diff':<10}")
    print("-" * 105)

    # Iterate through joints in URDF
    for joint in robot['joints']:
        joint_name = joint['name']
        
        # Skip base joint/fixed joints if not in MuJoCo joints, 
        # but in Spart, fixed joints are often just transforms.
        # URDF joints connect Parent -> Child.
        # "pos" in MuJoCo joint is relative to body frame. 
        # "pos" in MuJoCo body is relative to parent body.
        
        # In Spart/URDF: joint['T'] is the transform from Parent Link to Child Link (at q=0).
        # This includes the joint origin offset.
        
        # Let's find the corresponding MuJoCo Body (Child Link)
        # Identify child link name
        child_link_id = joint['child_link']
        # Find link in robot['links'] with this ID
        child_link = next((l for l in robot['links'] if l['id'] == child_link_id), None)
        if child_link is None: 
            continue
            
        child_link_name = child_link['name']
        
        # Find MuJoCo Body ID
        mj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_link_name)
        if mj_body_id == -1:
            # Try appending "_link" or removing it
            if child_link_name.endswith("_link"):
                alt_name = child_link_name[:-5]
                mj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, alt_name)
            else:
                mj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_link_name + "_link")
        
        if mj_body_id == -1:
            print(f"{child_link_name:<25} | Body not found in MuJoCo")
            continue
            
        # Get MuJoCo Body Pos and Quat (relative to parent)
        mj_pos = model.body_pos[mj_body_id]
        mj_quat = model.body_quat[mj_body_id] # [w, x, y, z]
        
        # Get Spart/URDF Pos and Quat from joint['T']
        # Note: Spart joint['T'] is from Parent CoM to Joint Frame.
        # We need to reconstruct Parent Link Frame to Joint Frame.
        # T_original = T_parent_link_to_inertial @ joint['T']
        
        parent_link_id = joint['parent_link']
        if parent_link_id == 0:
            # Base Link - Assuming Identity for Base Inertial Offset (or parsed elsewhere)
            # In SC_ur10e.urdf base inertial origin is 0 0 0.
            T_parent_inertial = np.eye(4)
        else:
            parent_link = next((l for l in robot['links'] if l['id'] == parent_link_id), None)
            T_parent_inertial = parent_link['T']
            
        T_urdf_reconstructed = T_parent_inertial @ joint['T']
        
        urdf_pos, urdf_quat = transform_to_pos_quat(T_urdf_reconstructed)
        
        # Compare Pos
        diff_pos = np.linalg.norm(mj_pos - urdf_pos)
        pos_str_mj = f"[{mj_pos[0]:.3f} {mj_pos[1]:.3f} {mj_pos[2]:.3f}]"
        pos_str_urdf = f"[{urdf_pos[0]:.3f} {urdf_pos[1]:.3f} {urdf_pos[2]:.3f}]"
        
        print(f"{child_link_name:<25} | {'Pos':<10} | {pos_str_mj:<25} | {pos_str_urdf:<25} | {diff_pos:.1e}")
        
        # Compare Quat
        # Quaternions are same if q or -q
        diff_quat = np.min([
            np.linalg.norm(mj_quat - urdf_quat),
            np.linalg.norm(mj_quat + urdf_quat)
        ])
        
        quat_str_mj = f"[{mj_quat[0]:.2f} {mj_quat[1]:.2f} {mj_quat[2]:.2f} {mj_quat[3]:.2f}]"
        quat_str_urdf = f"[{urdf_quat[0]:.2f} {urdf_quat[1]:.2f} {urdf_quat[2]:.2f} {urdf_quat[3]:.2f}]"
        
        print(f"{'':<25} | {'Quat':<10} | {quat_str_mj:<25} | {quat_str_urdf:<25} | {diff_quat:.1e}")
        
        # Check Joint Axis if it exists
        if joint['type'] in [1, 2]: # Revolute or Prismatic
            mj_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if mj_joint_id != -1:
                mj_axis = model.jnt_axis[mj_joint_id]
                urdf_axis = joint['axis']
                
                diff_axis = np.linalg.norm(mj_axis - urdf_axis)
                
                axis_str_mj = f"[{mj_axis[0]:.1f} {mj_axis[1]:.1f} {mj_axis[2]:.1f}]"
                axis_str_urdf = f"[{urdf_axis[0]:.1f} {urdf_axis[1]:.1f} {urdf_axis[2]:.1f}]"
                
                print(f"{'':<25} | {'Axis':<10} | {axis_str_mj:<25} | {axis_str_urdf:<25} | {diff_axis:.1e}")
            else:
                 print(f"{'':<25} | {'Axis':<10} | {'Not Found':<25} | {'-':<25} | -")
        
        print("-" * 105)

def test_robot_properties():
    print("========================================")
    print("STEP 1: Verify Robot Description (URDF vs XML)")
    print("========================================")

    # 1. Load XML
    xml_path = os.path.join(ROOT_DIR, 'assets/spacerobot_cjt.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    print(f"Loaded MuJoCo model from {xml_path}")

    # 2. Load URDF
    urdf_path = os.path.join(ROOT_DIR, 'assets/SC_ur10e.urdf') # Assuming this is the correct URDF
    try:
        # urdf2robot might need relative path or absolute
        robot, _ = urdf2robot(urdf_path)
        print(f"Loaded URDF robot from {urdf_path}")
    except Exception as e:
        # Fallback to src/dynamics/assets if not found
        urdf_path = os.path.join(ROOT_DIR, 'src/dynamics/assets/SC_ur10e.urdf')
        robot, _ = urdf2robot(urdf_path)
        print(f"Loaded URDF robot from {urdf_path}")

    # 3. Compare Base Link
    # MuJoCo base link
    # Body 1 is usually the first body after world (body 0)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_mass_mj = model.body_mass[base_body_id]
    base_inertia_mj = model.body_inertia[base_body_id] # This is diagonal if diaginertia is used
    
    print(f"\n[Base Link Check]")
    print(f"MuJoCo Mass: {base_mass_mj}")
    print(f"Spart Mass:  {robot['base_link']['mass']}")
    
    # Spart inertia is full 3x3. MuJoCo stores diagonal if diaginertia, or full in body_inertia? 
    # model.body_inertia is (nbody x 3). It stores diagonal elements of inertia frame.
    # To get full inertia in body frame, we need to consider body_quat (inertial frame orientation).
    # But usually for diagonal inertia, it matches if frame is aligned.
    print(f"MuJoCo Inertia (diag): {base_inertia_mj}")
    print(f"Spart Inertia (diag):  {np.diag(robot['base_link']['inertia'])}")

    # 4. Compare Links
    print(f"\n[Link Check]")
    # Get all link names from Spart
    for i, link in enumerate(robot['links']):
        # Find corresponding body in MuJoCo
        # Spart link names might match MuJoCo body names
        link_name = link['name']
        mj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        
        if mj_id == -1:
            # Try appending "_link" if not found, or handle naming diffs
            mj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name + "_link")
        
        if mj_id != -1:
            mj_mass = model.body_mass[mj_id]
            mj_inertia = model.body_inertia[mj_id]
            print(f"Link: {link_name:15s} | Mass (MJ/Sp): {mj_mass:.3f} / {link['mass']:.3f}")
        else:
            print(f"Link: {link_name:15s} | Not found in MuJoCo")

def test_dynamics():
    print("\n========================================")
    print("STEP 2: Verify Dynamics Matrices (H, M, C)")
    print("========================================")

    # 1. Setup Models
    xml_path = os.path.join(ROOT_DIR, 'assets/spacerobot_cjt.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    urdf_path = os.path.join(ROOT_DIR, 'assets/SC_ur10e.urdf')
    try:
        robot, _ = urdf2robot(urdf_path)
    except:
        urdf_path = os.path.join(ROOT_DIR, 'src/dynamics/assets/SC_ur10e.urdf')
        robot, _ = urdf2robot(urdf_path)

    n_q_joints = robot['n_q']
    
    # 2. Set Random State
    np.random.seed(42)
    # Joint angles
    q_joints = np.random.uniform(-1, 1, n_q_joints)
    # Joint velocities
    qdot_joints = np.random.uniform(-1, 1, n_q_joints)
    # Base Pose (pos, quat)
    base_pos = np.array([0, 0, 0])
    base_quat = np.array([1, 0, 0, 0]) # w, x, y, z for Spart? MuJoCo is w, x, y, z
    # Base Velocity (u0 for Spart = [w, v])
    # MuJoCo qvel for freejoint = [v, w] (linear world, angular body)
    base_lin_vel = np.random.uniform(-0.5, 0.5, 3)
    base_ang_vel = np.random.uniform(-0.5, 0.5, 3)
    
    # --- Set MuJoCo State ---
    # qpos: freejoint (7) + joints (6)
    # freejoint: pos(3), quat(4)
    data.qpos[0:3] = base_pos
    data.qpos[3:7] = base_quat
    data.qpos[7:] = q_joints
    
    # qvel: freejoint (6) + joints (6)
    # freejoint: linear(3), angular(3)
    data.qvel[0:3] = base_lin_vel
    data.qvel[3:6] = base_ang_vel
    data.qvel[6:] = qdot_joints
    
    mujoco.mj_forward(model, data)

    # --- Compute MuJoCo Mass Matrix ---
    # mj_fullM computes the dense mass matrix
    M_mj = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_mj, data.qM)
    
    # Compute C+G (Bias forces)
    # data.qfrc_bias contains Coriolis + Gravity + Centrifugal
    # To separate, we might need to zero gravity or something, but let's check total bias
    # But Spart separates C and G? 
    # Spart has `convective_inertia_matrix` (C). 
    # Gravity is not explicitly in `convective_inertia_matrix`. 
    # We can check Mass Matrix M first.

    # --- Set Spart State ---
    # Spart expects:
    # R0, r0: Base pos/orn
    # u0: Base vel [w, v] (angular body, linear inertial)
    # qm: Joint pos
    # um: Joint vel
    
    # Convert quat to R0
    # MuJoCo quat is [w, x, y, z]
    # Spart `quat_dcm` expects [x, y, z, w]? Let's check spart_functions.py
    # `q1, q2, q3, q0 = q` -> x, y, z, w
    # So we need to reorder MuJoCo quat [w, x, y, z] to [x, y, z, w] for Spart
    quat_spart = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
    R0 = ft.quat_dcm(quat_spart)
    r0 = base_pos.reshape(3, 1)
    
    # u0 = [w, v]
    # MuJoCo qvel[0:3] is v (linear inertial? "linear velocity in the global frame")
    # MuJoCo qvel[3:6] is w (angular body)
    u0 = np.concatenate([base_ang_vel, base_lin_vel]).reshape(6, 1)
    
    qm = q_joints.reshape(-1, 1)
    um = qdot_joints.reshape(-1, 1)
    
    # Compute Spart Dynamics
    RJ, RL, rJ, rL, e, g = ft.kinematics(R0, r0, qm, robot)
    Bij, Bi0, P0, pm = ft.diff_kinematics(R0, r0, rL, e, g, robot)
    t0, tL = ft.velocities(Bij, Bi0, P0, pm, u0, um, robot)
    
    I0, Im = ft.inertia_projection(R0, RL, robot)
    M0_tilde, Mm_tilde = ft.mass_composite_body(I0, Im, Bij, Bi0, robot)
    H0, H0m, Hm = ft.generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)
    
    # Construct Full Spart Mass Matrix
    # M_spart = [H0   H0m]
    #           [H0m.T Hm]
    # Dimensions: H0 (6x6), H0m (6xN), Hm (NxN)
    
    M_spart = np.block([
        [H0,    H0m],
        [H0m.T, Hm]
    ])
    
    # --- Compare M ---
    # MuJoCo M is ordered [v, w, q]
    # Spart M is ordered [w, v, q] (because u0 is [w, v])
    # We need to swap first 3 and next 3 rows/cols of M_spart to match MuJoCo
    
    # Permutation for 6x6 block
    # Spart: 0,1,2 (w), 3,4,5 (v)
    # MuJoCo: 0,1,2 (v), 3,4,5 (w)
    
    perm = [3, 4, 5, 0, 1, 2] + list(range(6, 6 + n_q_joints))
    M_spart_permuted = M_spart[np.ix_(perm, perm)]
    
    diff_M = np.abs(M_mj - M_spart_permuted)
    print(f"\n[Mass Matrix Comparison]")
    print(f"Max Difference: {np.max(diff_M)}")
    print(f"Mean Difference: {np.mean(diff_M)}")
    
    if np.max(diff_M) < 1e-4:
        print(">> Mass Matrix Matches!")
    else:
        print(">> Mass Matrix Mismatch.")
        # print("MuJoCo M (top-left 6x6):\n", M_mj[:6,:6])
        # print("Spart M (top-left 6x6 permuted):\n", M_spart_permuted[:6,:6])

    # --- Compare Coriolis/Centrifugal ---
    # Spart: C0, C0m, Cm0, Cm
    # C * u = bias force (excluding gravity?)
    # ft.convective_inertia_matrix returns matrices C
    # Vector C(q, u) * u
    
    C0, C0m, Cm0, Cm = ft.convective_inertia_matrix(t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)
    
    # Construct Full C matrix
    C_spart = np.block([
        [C0,    C0m],
        [Cm0,   Cm]
    ])
    
    # Compute C * u_vector
    # u_vec_spart = [u0; um] = [w; v; qdot]
    u_vec_spart = np.concatenate([u0, um])
    
    coriolis_force_spart = C_spart @ u_vec_spart
    
    # Permute Coriolis Force to match MuJoCo [v; w; qdot]
    # swap first 3 and next 3 elements
    coriolis_force_spart_perm = np.concatenate([
        coriolis_force_spart[3:6],
        coriolis_force_spart[0:3],
        coriolis_force_spart[6:]
    ])
    
    # MuJoCo Coriolis + Centrifugal + Gravity
    # data.qfrc_bias
    # To isolate C*u, we should set gravity to zero in MuJoCo?
    # Or subtract Gravity?
    
    # Let's set gravity to zero in model for this check
    model.opt.gravity[:] = 0
    mujoco.mj_forward(model, data)
    
    bias_mj = data.qfrc_bias # This is now just C(q,qdot)*qdot
    
    # Flatten spart vector to avoid broadcasting errors (12,) vs (12,1)
    diff_C = np.abs(bias_mj - coriolis_force_spart_perm.flatten())
    
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    
    print("\n" + "="*60)
    print(" DETAILED DYNAMICS COMPARISON (H, M, C)")
    print("="*60)
    
    print("\n[Mass/Inertia Matrix (M/H)]")
    print("-" * 30)
    print("MuJoCo M (First 6x6 - Base):")
    print(M_mj[:6, :6])
    print("\nSpart M (First 6x6 - Base):")
    print(M_spart_permuted[:6, :6])
    
    print("\nMuJoCo M (Joints):")
    print(M_mj[6:, 6:])
    print("\nSpart M (Joints):")
    print(M_spart_permuted[6:, 6:])
    
    print("\n[Coriolis/Centrifugal Vector (C*u)]")
    print("-" * 30)
    print(f"MuJoCo Bias: {bias_mj}")
    print(f"Spart Bias:  {coriolis_force_spart_perm.flatten()}")
    
    print(f"\n[Summary]")
    print(f"Mass Matrix Max Diff: {np.max(diff_M):.6f}")
    print(f"Coriolis Force Max Diff: {np.max(diff_C):.6f}")
    
    if np.max(diff_C) < 1e-3: # Slightly looser tolerance for C
        print(">> Coriolis Forces Match!")
    else:
        print(">> Coriolis Forces Mismatch.")

if __name__ == "__main__":
    # test_robot_properties()
    # test_kinematics()
    test_dynamics()
