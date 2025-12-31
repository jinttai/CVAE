import torch

"""
t0 : [6x1] Base-link twist projected in the inertial coordinate system (CCS). The first three elements represent the angular velocity, and the last three represent the linear velocity.
tL : [6xn] Twists of the links projected in the inertial CCS.
P0 : [6x6] Base-link twist-propagation matrix, which transforms base-link velocities into generalized velocities.
pm : [6xn] Manipulator twist-propagation matrix, defining the effect of joint velocities on the system’s motion.
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
g : [3xn] Vector from the ith joint’s origin to the ith link’s origin, projected in the inertial CCS.
Bij : [6x6xn] Twist-propagation matrix for manipulator links.
Bi0 : [6x6xn] Twist-propagation matrix between the base and the links.
I0 : [3x3] Inertia matrix of the base-link, projected in the inertial CCS.
Im : [3x3xn] Inertia matrices of the links, projected in the inertial CCS.
M0_tilde : [6x6] Composite mass matrix of the base-link.
Mm_tilde : [6x6xn] Composite mass matrices of the manipulator links.
H0 : [6x6] Base-link inertia matrix.
H0m : [6xn_q] Base-link to manipulator coupling inertia matrix.
Hm : [n_qxn_q] Manipulator inertia matrix.
C0 : [6x6] Base-link convective inertia matrix.
C0m : [6xn_q] Coupling convective inertia matrix between base-link and manipulator.
Cm0 : [n_qx6] Coupling convective inertia matrix between manipulator and base-link.
Cm : [n_qxn_q] Manipulator convective inertia matrix.
J0 : [6x6] Geometric Jacobian of the base-link.
Jm : [6xn_q] Geometric Jacobian of the manipulator.
J0dot : [6x6] Time derivative of the base-link Jacobian.
Jmdot : [6xn_q] Time derivative of the manipulator Jacobian.
N : [(6+6*n)x(6+n_q)] Natural Orthogonal Complement (NOC) matrix.
Ndot : [(6+6*n)x(6+n_q)] Time derivative of the NOC matrix.
r_com : [3x1] Center of mass (CoM) position of the entire system, projected in the inertial CCS.
"""


def skew_symmetric(v):
    # Ensure v is a tensor on the correct device/dtype
    v = torch.as_tensor(v)
    device = v.device
    dtype = v.dtype
    v = v.flatten()
    assert len(v) == 3, 'vector length error'
    
    # Use torch.stack to preserve gradients
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    return torch.stack([
        torch.stack([zero, -v[2], v[1]]),
        torch.stack([v[2], zero, -v[0]]),
        torch.stack([-v[1], v[0], zero])
    ])

def euler_dcm(e, alpha):
    e = torch.as_tensor(e)
    alpha = torch.as_tensor(alpha)
    # alpha can be a scalar with grad, e is usually a constant axis vector
    # preserve device/dtype from inputs
    
    q = torch.hstack((e * torch.sin(alpha / 2), torch.cos(alpha / 2)))
    return quat_dcm(q)

def quat_dcm(q):
    q = torch.as_tensor(q)
    device = q.device
    dtype = q.dtype
    q = q.flatten()
    assert len(q) == 4, 'quaternion length error'
    q1, q2, q3, q0 = q
    
    # Use torch.stack to preserve gradients
    # [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)]
    # ...
    
    one = torch.tensor(1.0, device=device, dtype=dtype)
    two = torch.tensor(2.0, device=device, dtype=dtype)
    
    r1 = torch.stack([
        one - two * (q2**2 + q3**2), 
        two * (q1 * q2 - q0 * q3), 
        two * (q1 * q3 + q0 * q2)
    ])
    r2 = torch.stack([
        two * (q1 * q2 + q0 * q3), 
        one - two * (q1**2 + q3**2), 
        two * (q2 * q3 - q0 * q1)
    ])
    r3 = torch.stack([
        two * (q1 * q3 - q0 * q2), 
        two * (q2 * q3 + q0 * q1), 
        one - two * (q1**2 + q2**2)
    ])
    
    return torch.stack([r1, r2, r3])

def quat_dot(q, w):
    q = torch.as_tensor(q)
    w = torch.as_tensor(w)
    device = q.device
    dtype = q.dtype
    
    q = q.flatten()
    w = w.flatten()
    assert len(q) == 4 and len(w) == 3, 'quaternion or angular velocity length error'
    q1, q2, q3, q0 = q
    w1, w2, w3 = w
    
    half = torch.tensor(0.5, device=device, dtype=dtype)
    
    return torch.stack([
        -half * (w1 * q2 + w2 * q3 + w3 * q0),
        half * (w1 * q0 - w2 * q3 + w3 * q2),
        half * (w2 * q0 + w1 * q3 - w3 * q1),
        half * (w3 * q0 - w1 * q2 + w2 * q1)
    ])

def accelerations(t0, tL, P0, pm, Bi0, Bij, u0, um, u0dot, umdot, robot):
    n = robot['n_links_joints']
    
    # Extract device/dtype from t0
    device = t0.device
    dtype = t0.dtype
    
    # Omega0 block
    Omega0 = torch.zeros((6, 6), device=device, dtype=dtype)
    Omega0[:3, :3] = skew_symmetric(t0[:3])
    
    Omegam_list = []
    for i in range(n):
        skew_tLi = skew_symmetric(tL[:3, i])
        # Omegam[:, :, i] block
        # Create block [[skew, 0], [0, skew]]
        zeros = torch.zeros((3, 3), device=device, dtype=dtype)
        Omegam_i = torch.cat([
            torch.cat([skew_tLi, zeros], dim=1),
            torch.cat([zeros, skew_tLi], dim=1)
        ], dim=0)
        Omegam_list.append(Omegam_i)
        
    Omegam = torch.stack(Omegam_list, dim=2)
        
    t0dot = Omega0 @ P0 @ u0 + P0 @ u0dot
    
    tLdot_list = []
    for i in range(n):
        parent_link = robot['joints'][i]['parent_link']
        if parent_link == 0:
            skew_diff = skew_symmetric(t0[3:6].flatten() - tL[3:6, i])
            # Construct block for calculation
            # [[0, 0], [skew_diff, 0]]
            zeros = torch.zeros((3, 3), device=device, dtype=dtype)
            block_mat = torch.cat([
                torch.cat([zeros, zeros], dim=1),
                torch.cat([skew_diff, zeros], dim=1)
            ], dim=0)
            
            val = (Bi0[:, :, i] @ t0dot + block_mat @ t0).flatten()
        else:
            skew_diff = skew_symmetric(tL[3:6, parent_link-1] - tL[3:6, i])
            zeros = torch.zeros((3, 3), device=device, dtype=dtype)
            block_mat = torch.cat([
                torch.cat([zeros, zeros], dim=1),
                torch.cat([skew_diff, zeros], dim=1)
            ], dim=0)
            
            # tLdot_list has previous values. parent_link-1 < i if sequential.
            # But checking urdf2robot: parent is visited before child.
            # parent_link is index in LINKS. joint i connects parent_link to child_link i+1.
            # parent_link index matches joint index of the joint that created it?
            # No. robot['links'] has n items (excluding base).
            # Link 0 is connected by Joint 0 (id 1).
            # parent_link index: 0 means base. 1..n means link indices.
            # tLdot_list is indexed 0..n-1.
            # We need tLdot from parent.
            
            parent_idx = parent_link - 1
            tLdot_parent = tLdot_list[parent_idx]
            tL_parent = tL[:, parent_idx]
            
            val = (Bij[:, :, i, parent_idx] @ tLdot_parent + 
                   block_mat @ tL_parent).flatten()
        
        if robot['joints'][i]['type'] != 0:
            q_id = robot['joints'][i]['q_id']
            # pm[:, i] is vector (6,)
            # Omegam_list[i] is (6,6)
            val = val + Omegam_list[i] @ pm[:, i] * um[q_id-1] + pm[:, i] * umdot[q_id-1]
            
        tLdot_list.append(val)
        
    tLdot = torch.stack(tLdot_list, dim=1)
    return t0dot, tLdot

def center_of_mass(r0, rL, robot):
    # device/dtype from r0
    device = r0.device
    dtype = r0.dtype
    
    mass_total = torch.as_tensor(robot['base_link']['mass'], device=device, dtype=dtype)
    mass_r = r0 * robot['base_link']['mass']
    
    for i in range(robot['n_links_joints']):
        mass_i = torch.as_tensor(robot['links'][i]['mass'], device=device, dtype=dtype)
        mass_total = mass_total + mass_i
        mass_r = mass_r + rL[:, i].reshape(3,1) * mass_i
        
    return mass_r / mass_total

def kinematics(R0, r0, qm, robot):
    n = robot['n_links_joints']
    device = R0.device
    dtype = R0.dtype
    
    # T0 block [[R0, r0], [0, 1]]
    # T0 = torch.eye(4, device=device, dtype=dtype)
    # T0[:3, :3] = R0
    # T0[:3, 3] = r0.flatten()
    # Avoid in-place
    row4 = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).reshape(1, 4)
    T0 = torch.cat([
        torch.cat([R0, r0.reshape(3, 1)], dim=1),
        row4
    ], dim=0)

    TJ_list = [None] * n
    TL_list = [None] * n
    
    for i in range(n):
        joint = robot['joints'][i]
        
        # Ensure joint['T'] is on correct device
        joint_T = joint['T'].to(device=device, dtype=dtype)
        
        if joint['parent_link'] == 0:
            TJ_val = T0 @ joint_T
        else:
            # parent_link is 1-based index for links list?
            # urdf2robot: robot_keys['link_id'][base_link_name] = 0.
            # child_link: link_id + 1.
            # So link IDs are 1..n.
            # TL_list is 0..n-1 corresponding to link 1..n.
            # parent_link - 1 gives index in TL_list.
            TJ_val = TL_list[joint['parent_link'] - 1] @ joint_T
            
        TJ_list[i] = TJ_val
        
        T_qm = torch.eye(4, device=device, dtype=dtype)
        
        if joint['type'] == 1:
            # T_qm block
            axis = joint['axis'].to(device=device, dtype=dtype)
            # Easiest way to construct T_qm without inplace:
            # Use the rotation matrix and manually build 4x4
            R_qm = euler_dcm(axis, qm[joint['q_id'] - 1])
            zeros = torch.zeros((3, 1), device=device, dtype=dtype)
            
            # [[R, 0], [0, 1]]
            T_qm = torch.cat([
                torch.cat([R_qm, zeros], dim=1),
                row4
            ], dim=0)
            
        elif joint['type'] == 2:
            # joint['axis'].reshape(3,1) * qm[...]
            axis = joint['axis'].to(device=device, dtype=dtype)
            p_qm = (axis.flatten() * qm[joint['q_id'] - 1]).reshape(3, 1)
            I3 = torch.eye(3, device=device, dtype=dtype)
            
            T_qm = torch.cat([
                torch.cat([I3, p_qm], dim=1),
                row4
            ], dim=0)
        else:
            pass # Identity
            
        link = robot['links'][joint['child_link'] - 1]
        link_T = link['T'].to(device=device, dtype=dtype)
        
        # TJ_list[i] corresponds to joint['id']-1?
        # joint['id'] is i+1. So yes.
        # link['parent_joint'] should be i+1.
        
        TL_val = TJ_val @ T_qm @ link_T
        TL_list[link['id'] - 1] = TL_val
    
    # Stack lists to create tensors
    TJ = torch.stack(TJ_list, dim=2)
    TL = torch.stack(TL_list, dim=2)
    
    RJ = TJ[:3, :3, :]
    RL = TL[:3, :3, :]
    rJ = TJ[:3, 3, :]
    rL = TL[:3, 3, :]
    
    e_list = []
    g_list = []
    
    for i in range(n):
        axis = robot['joints'][i]['axis'].to(device=device, dtype=dtype)
        e_val = RJ[:, :, i] @ axis
        
        # rL[:, i] is rL_i
        # rJ parent: robot['links'][i]['parent_joint'] - 1
        # parent_joint of link i is joint i+1 (id i+1).
        # So parent_joint index is i.
        # Wait, g vector definition: rL_i - rJ_i.
        # Code says: rL[:, i] - rJ[:, robot['links'][i]['parent_joint'] - 1]
        
        # Check indices:
        # link i has parent joint.
        pj_idx = robot['links'][i]['parent_joint'] - 1
        g_val = rL[:, i] - rJ[:, pj_idx]
        
        e_list.append(e_val)
        g_list.append(g_val)
        
    e = torch.stack(e_list, dim=1)
    g = torch.stack(g_list, dim=1)
        
    return RJ, RL, rJ, rL, e, g

def diff_kinematics(R0, r0, rL, e, g, robot):
    """
    Vectorized diff_kinematics - O(n^2) loop removed
    """
    n = robot['n_links_joints']
    device = R0.device
    dtype = R0.dtype
    
    # 1. P0 (unchanged)
    zeros_33 = torch.zeros((3, 3), device=device, dtype=dtype)
    I3 = torch.eye(3, device=device, dtype=dtype)
    P0 = torch.cat([
        torch.cat([R0, zeros_33], dim=1),
        torch.cat([zeros_33, I3], dim=1)
    ], dim=0)
    
    # 2. Vectorized Bij calculation
    # rL: [3, n] -> rL_diff[k, i, j] = rL[k, j] - rL[k, i]
    # We want skew(rL[:, j] - rL[:, i]) for Bij[i][j]
    
    rL_j = rL.unsqueeze(2)  # [3, n, 1] -> rL[:, i] when broadcasted at index i
    rL_i = rL.unsqueeze(1)  # [3, 1, n] -> rL[:, j] when broadcasted at index j
    
    # rL_diff[k, i, j] should be rL[k, j] - rL[k, i]
    # rL_i broadcasts to [3, n, n] such that at (i, j) it gives rL[:, j]
    # rL_j broadcasts to [3, n, n] such that at (i, j) it gives rL[:, i]
    
    rL_diff = rL_i - rL_j   # [3, n, n] (j - i)
    
    # Skew symmetric for all pairs
    # rL_diff is [3, n, n] -> [x, y, z] components are [n, n] matrices
    # Skew(v) = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    
    # [Fix for vmap inplace error]
    # Do not use inplace assignment like skew_all[0, 1] = ...
    # Instead, construct rows and stack them.
    
    # Components of rL_diff
    rx = rL_diff[0] # [n, n]
    ry = rL_diff[1]
    rz = rL_diff[2]
    
    zeros_nn = torch.zeros((n, n), device=device, dtype=dtype)
    
    # Row 0: [0, -rz, ry]
    row0 = torch.stack([zeros_nn, -rz, ry], dim=0) # [3, n, n]
    
    # Row 1: [rz, 0, -rx]
    row1 = torch.stack([rz, zeros_nn, -rx], dim=0) # [3, n, n]
    
    # Row 2: [-ry, rx, 0]
    row2 = torch.stack([-ry, rx, zeros_nn], dim=0) # [3, n, n]
    
    # Stack rows -> [3, 3, n, n]
    skew_all = torch.stack([row0, row1, row2], dim=0)
    
    # Bij block matrix construction [[I, 0], [skew, I]]
    # Expand I3 to [3, 3, n, n]
    I3_expand = I3.unsqueeze(-1).unsqueeze(-1).expand(3, 3, n, n)
    zeros_expand = torch.zeros((3, 3, n, n), device=device, dtype=dtype)
    
    # Bij = torch.zeros((6, 6, n, n), device=device, dtype=dtype)
    # Using slice assignment might be faster or cat
    # Bij[:3, :3] = I3_expand
    # Bij[:3, 3:] = zeros_expand
    # Bij[3:, :3] = skew_all
    # Bij[3:, 3:] = I3_expand
    
    # Concatenation approach
    # Row 1: [I3, 0]
    row1 = torch.cat([I3_expand, zeros_expand], dim=1) # [3, 6, n, n]
    # Row 2: [skew, I3]
    row2 = torch.cat([skew_all, I3_expand], dim=1) # [3, 6, n, n]
    Bij = torch.cat([row1, row2], dim=0) # [6, 6, n, n]
    
    # Apply branch mask
    # robot['con']['branch'] is [n, n]
    branch_mask = robot['con']['branch'].to(dtype=dtype)
    # Broadcast: [1, 1, n, n] * [n, n]
    Bij = Bij * branch_mask.unsqueeze(0).unsqueeze(0)
    
    # 3. Vectorized Bi0 calculation
    # r0_diff = r0 - rL_i => r0 [3] - rL [3, n]
    r0_expand = r0.flatten().unsqueeze(1) # [3, 1]
    r0_diff = r0_expand - rL # [3, n]
    
    # [Fix for vmap inplace error]
    rx0 = r0_diff[0] # [n]
    ry0 = r0_diff[1]
    rz0 = r0_diff[2]
    
    zeros_n_vec = torch.zeros(n, device=device, dtype=dtype)
    
    # Row 0: [0, -rz, ry]
    row0_Bi0 = torch.stack([zeros_n_vec, -rz0, ry0], dim=0) # [3, n]
    
    # Row 1: [rz, 0, -rx]
    row1_Bi0 = torch.stack([rz0, zeros_n_vec, -rx0], dim=0) # [3, n]
    
    # Row 2: [-ry, rx, 0]
    row2_Bi0 = torch.stack([-ry0, rx0, zeros_n_vec], dim=0) # [3, n]
    
    skew_Bi0 = torch.stack([row0_Bi0, row1_Bi0, row2_Bi0], dim=0) # [3, 3, n]
    
    I3_n = I3.unsqueeze(-1).expand(3, 3, n)
    zeros_n = torch.zeros((3, 3, n), device=device, dtype=dtype)
    
    # Bi0 = [[I, 0], [skew, I]]
    Bi0_row1 = torch.cat([I3_n, zeros_n], dim=1) # [3, 6, n]
    Bi0_row2 = torch.cat([skew_Bi0, I3_n], dim=1) # [3, 6, n]
    Bi0 = torch.cat([Bi0_row1, Bi0_row2], dim=0) # [6, 6, n]
    
    # 4. pm calculation
    # Vectorized approach hard because of conditional logic based on joint type?
    # Joint types are in a list, not tensor. But we can iterate or mask.
    # n is usually small (e.g., 12-20). Loop might be fine, but we can vectorize if we gather types.
    
    # Since n is small, let's keep loop for pm or use simple masking if all revolute.
    # A1 robot is all revolute (type 1).
    # General solution:
    
    # [Fix for vmap inplace error]
    # Instead of pm[:, i] = ..., use a list and stack.
    
    # Vectorized cross product
    cross_eg = torch.linalg.cross(e, g, dim=0) # [3, n]
    
    pm_list = []
    
    for i in range(n):
        jt = robot['joints'][i]['type']
        if jt == 1: # Revolute
            # pm_i = [e[:, i], cross_eg[:, i]]
            pm_i = torch.cat([e[:, i], cross_eg[:, i]], dim=0)
        elif jt == 2: # Prismatic
            # pm_i = [zeros(3), e[:, i]]
            zeros_3 = torch.zeros(3, device=device, dtype=dtype)
            pm_i = torch.cat([zeros_3, e[:, i]], dim=0)
        else:
            pm_i = torch.zeros(6, device=device, dtype=dtype)
        
        pm_list.append(pm_i)
    
    pm = torch.stack(pm_list, dim=1) # [6, n]
            
    return Bij, Bi0, P0, pm

def velocities(Bij, Bi0, P0, pm, u0, um, robot):
    n = robot['n_links_joints']
    device = P0.device
    dtype = P0.dtype
    
    t0 = P0 @ u0
    tL_list = []
    for i in range(n):
        parent_link = robot['joints'][i]['parent_link']
        if parent_link == 0:
            val = (Bi0[:, :, i] @ t0).flatten()
        else:
            # Bij[:, :, i, i - 1]
            # i-1 is index of parent link in tL list
            # i is index of current link
            # check indices. Bij is (6,6,n,n). Bij[:,:,child,parent]
            val = Bij[:, :, i, i - 1] @ tL_list[i - 1]
            
        if robot['joints'][i]['type'] != 0:
            q_id = robot['joints'][i]['q_id']
            val = val + pm[:, i] * um[q_id-1]
        tL_list.append(val)
        
    tL = torch.stack(tL_list, dim=1)
    return t0, tL

def inertia_projection(R0, RL, robot):
    device = R0.device
    dtype = R0.dtype
    
    base_inertia = robot['base_link']['inertia'].to(device=device, dtype=dtype)
    I0 = R0 @ base_inertia @ R0.T
    
    n = robot['n_links_joints']
    Im_list = []
    for i in range(n):
        link_inertia = robot['links'][i]['inertia'].to(device=device, dtype=dtype)
        # RL[:, :, i] is (3, 3)
        val = RL[:, :, i] @ link_inertia @ RL[:, :, i].T
        Im_list.append(val)
        
    Im = torch.stack(Im_list, dim=2)
    return I0, Im

def mass_composite_body(I0, Im, Bij, Bi0, robot):
    n = robot['n_links_joints']
    device = I0.device
    dtype = I0.dtype
    
    # Mm_tilde = torch.zeros((6, 6, n), device=device, dtype=dtype)
    # Since we fill in reverse, we can't append.
    # Pre-allocate a list of None, then stack.
    # But updating happens in reverse and depends on children.
    # Mm_tilde[:, :, i] += ... where children j > i.
    # So when computing i, we need values for j which are already computed.
    
    Mm_tilde_list = [None] * n
    
    for i in reversed(range(n)):
        # Mm_tilde block [[Im, 0], [0, m*I]]
        Im_i = Im[:, :, i]
        mass_i = torch.as_tensor(robot['links'][i]['mass'], device=device, dtype=dtype)
        zeros = torch.zeros((3, 3), device=device, dtype=dtype)
        I3 = torch.eye(3, device=device, dtype=dtype)
        
        # Initial value
        val = torch.cat([
            torch.cat([Im_i, zeros], dim=1),
            torch.cat([zeros, mass_i * I3], dim=1)
        ], dim=0)
        
        children = torch.nonzero(robot['con']['child'][:, i] == 1, as_tuple=True)[0]
        for j in children:
            # j > i guaranteed by traversal order? Usually yes.
            # Bij[:, :, j, i].T @ Mm_tilde[:, :, j] @ Bij[:, :, j, i]
            # Mm_tilde_list[j] should be valid.
            
            term = Bij[:, :, j, i].T @ Mm_tilde_list[j] @ Bij[:, :, j, i]
            val = val + term
            
        Mm_tilde_list[i] = val
            
    Mm_tilde = torch.stack(Mm_tilde_list, dim=2)
            
    # M0_tilde block [[I0, 0], [0, m*I]]
    mass_base = torch.as_tensor(robot['base_link']['mass'], device=device, dtype=dtype)
    zeros = torch.zeros((3, 3), device=device, dtype=dtype)
    I3 = torch.eye(3, device=device, dtype=dtype)
    
    M0_tilde = torch.cat([
        torch.cat([I0, zeros], dim=1),
        torch.cat([zeros, mass_base * I3], dim=1)
    ], dim=0)
    
    children = torch.nonzero(robot['con']['child_base'] == 1, as_tuple=True)[0]
    for j in children:
        term = Bi0[:, :, j].T @ Mm_tilde_list[j] @ Bi0[:, :, j]
        M0_tilde = M0_tilde + term
        
    return M0_tilde, Mm_tilde

def generalized_inertia_matrix_old(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot):
    n_q = robot['n_q']
    n = robot['n_links_joints']
    device = M0_tilde.device
    dtype = M0_tilde.dtype
    
    H0 = P0.T @ M0_tilde @ P0
    
    # Hm is n_q x n_q. Since n_q is small, we can use list of lists or just fill if we are careful.
    # But filling Hm[qi, qj] = ... is in-place.
    # Better to create a list of rows, then stack.
    # But indices are sparse/random (qi, qj).
    
    # Construct Hm by creating zero tensor and adding terms?
    # Adding to zero tensor: Hm = Hm + delta_matrix?
    # Delta matrix is mostly zeros except at qi, qj.
    # This is expensive for large matrices but fine for small n_q.
    
    Hm = torch.zeros((n_q, n_q), device=device, dtype=dtype)
    
    # Collecting terms in a dictionary might be better? {(qi, qj): val}
    terms = {}
    
    for j in range(n):
        for i in range(j, n):
            if robot['joints'][i]['type'] != 0 and robot['joints'][j]['type'] != 0:
                qi = robot['joints'][i]['q_id'] - 1
                qj = robot['joints'][j]['q_id'] - 1
                if qi >= 0 and qj >= 0:
                    val = (pm[:6, i] 
                            @ Mm_tilde[:6, :6, i] 
                            @ Bij[:6, :6, i, j] 
                            @ pm[:6, j])
                    
                    # Accumulate if multiple paths? Unlikely for tree.
                    terms[(qi, qj)] = val
                    if qi != qj:
                        terms[(qj, qi)] = val

    # Now construct Hm from terms.
    # Since we need gradients flow through val to input, we can't just use tensor[i,j] = val.
    # We can stack rows.
    rows = []
    for r in range(n_q):
        cols = []
        for c in range(n_q):
            if (r, c) in terms:
                cols.append(terms[(r, c)].reshape(1))
            else:
                cols.append(torch.zeros(1, device=device, dtype=dtype))
        rows.append(torch.cat(cols, dim=0))
    Hm = torch.stack(rows, dim=0)

    # H0m is 6 x n_q
    H0m_cols = []
    # n_q cols. Need to map q_id to col index.
    # robot['joints'][i]['q_id'] gives 1..n_q.
    # We iterate 1..n_q to build columns?
    # Or we iterate joints and place them.
    # We can build a dict {q_id: vec}
    
    h0m_dict = {}
    for i in range(n):
        if robot['joints'][i]['type'] != 0:
            qi = robot['joints'][i]['q_id'] - 1
            vec = pm[:6, i] @ Mm_tilde[:6, :6, i] @ Bi0[:6, :6, i] @ P0
            h0m_dict[qi] = vec # vec is (6,) or (1,6)? vec is (6,)
            
    cols = []
    for k in range(n_q):
        if k in h0m_dict:
            cols.append(h0m_dict[k])
        else:
            cols.append(torch.zeros(6, device=device, dtype=dtype))
            
    if cols:
        H0m = torch.stack(cols, dim=1)
    else:
        H0m = torch.zeros((6, n_q), device=device, dtype=dtype)
        
    return H0, H0m, Hm

def generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot):
    """
    Vectorized generalized_inertia_matrix - vmap compatible
    """
    n_q = robot['n_q']
    n = robot['n_links_joints']
    device = M0_tilde.device
    dtype = M0_tilde.dtype
    
    H0 = P0.T @ M0_tilde @ P0
    
    # Pre-compute q_id mappings for active joints
    active_indices = []
    q_ids = []
    
    for i in range(n):
        if robot['joints'][i]['type'] != 0:
            active_indices.append(i)
            q_ids.append(robot['joints'][i]['q_id'] - 1)
            
    num_active = len(active_indices)
    
    # If no active joints, return zeros
    if num_active == 0:
        return H0, torch.zeros((6, n_q), device=device, dtype=dtype), torch.zeros((n_q, n_q), device=device, dtype=dtype)
    
    active_idx_tensor = torch.tensor(active_indices, device=device, dtype=torch.long)
    # q_idx_tensor not needed for loop construction, but good for reference
    
    # Gather pm columns: [6, num_active]
    pm_active = pm[:, active_idx_tensor] 
    
    # Gather Mm_tilde blocks: [6, 6, num_active]
    Mm_tilde_active = Mm_tilde[:, :, active_idx_tensor]
    
    # Gather Bij blocks: [6, 6, num_active, num_active]
    grid_i, grid_j = torch.meshgrid(active_idx_tensor, active_idx_tensor, indexing='ij')
    Bij_active = Bij[:, :, grid_i, grid_j]
    
    # Hm calculation using einsum
    Mm_pm = torch.einsum('mnk,nk->mk', Mm_tilde_active, pm_active)
    Bij_pm = torch.einsum('mnij,nj->mij', Bij_active, pm_active)
    Hm_dense = torch.einsum('mi,mij->ij', Mm_pm, Bij_pm)
    
    # Symmetrize
    diagonal = torch.diagonal(Hm_dense)
    Hm_dense_sym = Hm_dense + Hm_dense.T - torch.diag(diagonal)
    
    # === vmap-compatible Hm construction ===
    # Check if q_ids are contiguous 0..n_q-1
    is_contiguous = (num_active == n_q)
    if is_contiguous:
        for i in range(num_active):
            if q_ids[i] != i:
                is_contiguous = False
                break
    
    if is_contiguous:
        # q_ids are [0, 1, 2, ..., n_q-1] in order - direct use
        Hm = Hm_dense_sym
    else:
        # Need to scatter - build using loops (vmap-safe since no in-place)
        Hm_rows = []
        for r in range(n_q):
            Hm_cols = []
            for c in range(n_q):
                # Find if (r, c) maps to any (q_ids[i], q_ids[j])
                val = torch.zeros(1, device=device, dtype=dtype)
                
                # Manual search to avoid advanced indexing inside loop which might confuse vmap
                # But since q_ids is constant per robot structure, we could optimize this outside vmap?
                # robot struct is passed in.
                
                found = False
                for i_idx, qi in enumerate(q_ids):
                    if qi == r:
                        for j_idx, qj in enumerate(q_ids):
                            if qj == c:
                                val = Hm_dense_sym[i_idx, j_idx].unsqueeze(0)
                                found = True
                                break
                    if found: break
                
                Hm_cols.append(val)
            Hm_rows.append(torch.cat(Hm_cols, dim=0))
        Hm = torch.stack(Hm_rows, dim=0)
    
    # === vmap-compatible H0m construction ===
    Bi0_active = Bi0[:, :, active_idx_tensor]
    term1 = torch.einsum('mnk,mk->nk', Bi0_active, Mm_pm)
    H0m_dense = P0.T @ term1  # [6, num_active]
    
    if is_contiguous:
        # Direct use
        H0m = H0m_dense
    else:
        # Build column by column
        H0m_cols = []
        for c in range(n_q):
            found = False
            for i_idx, qi in enumerate(q_ids):
                if qi == c:
                    H0m_cols.append(H0m_dense[:, i_idx])
                    found = True
                    break
            if not found:
                H0m_cols.append(torch.zeros(6, device=device, dtype=dtype))
        H0m = torch.stack(H0m_cols, dim=1)
    
    return H0, H0m, Hm

def convective_inertia_matrix(t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot):
    n_q = robot['n_q']
    n = robot['n_links_joints']
    device = t0.device
    dtype = t0.dtype
    
    # Omega0 block
    Omega0 = torch.zeros((6, 6), device=device, dtype=dtype)
    Omega0[:3, :3] = skew_symmetric(t0[:3])
    
    Omega_list = []
    zeros = torch.zeros((3, 3), device=device, dtype=dtype)
    for i in range(n):
        skew_tLi = skew_symmetric(tL[:3, i])
        Omega_i = torch.cat([
            torch.cat([skew_tLi, zeros], dim=1),
            torch.cat([zeros, skew_tLi], dim=1)
        ], dim=0)
        Omega_list.append(Omega_i)
    Omega = torch.stack(Omega_list, dim=2)
        
    # Mdot0 block
    zeros_66 = torch.zeros((6, 6), device=device, dtype=dtype)
    Mdot0 = torch.cat([
        torch.cat([Omega0[:3,:3] @ I0, zeros], dim=1),
        torch.cat([zeros, zeros], dim=1)
    ], dim=0)
    
    Mdot_list = []
    for i in range(n):
        val = torch.cat([
            torch.cat([Omega_list[i][:3, :3] @ Im[:, :, i], zeros], dim=1),
            torch.cat([zeros, zeros], dim=1)
        ], dim=0)
        Mdot_list.append(val)
    # Stack? Mdot is (6,6,n)
    # We need list access for Mdot_tilde computation
    
    Mdot_tilde_list = [None] * n
    for i in reversed(range(n)):
        val = Mdot_list[i]
        children = torch.nonzero(robot['con']['child'][:, i] == 1, as_tuple=True)[0]
        for j in children:
            val = val + Mdot_tilde_list[j]
        Mdot_tilde_list[i] = val
    
    Mdot_tilde = torch.stack(Mdot_tilde_list, dim=2)

    Mdot0_tilde = Mdot0
    children = torch.nonzero(robot['con']['child_base'] == 1, as_tuple=True)[0]
    for j in children:
        Mdot0_tilde = Mdot0_tilde + Mdot_tilde_list[j]

    Bdotij_list = [[None]*n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            if robot['con']['branch'][i, j] == 1:
                # Bdotij block
                skew_val = skew_symmetric(tL[3:6, j] - tL[3:6, i])
                val = torch.cat([
                    torch.cat([zeros, zeros], dim=1),
                    torch.cat([skew_val, zeros], dim=1)
                ], dim=0)
                Bdotij_list[i][j] = val
            else:
                Bdotij_list[i][j] = zeros_66

    # Hij_tilde
    Hij_tilde_list = [[None]*n for _ in range(n)]
    for i in reversed(range(n)):
        for j in reversed(range(n)):
            # Mm_tilde[:, :, i] @ Bdotij[:, :, i, j]
            Bdotij_val = Bdotij_list[i][j]
            val = Mm_tilde[:, :, i] @ Bdotij_val
            
            children = torch.nonzero(robot['con']['child'][:, i] == 1, as_tuple=True)[0]
            for k in children:
                # Bij[:, :, k, i].T @ Hij_tilde[:, :, k, i]
                # Wait, second index is j? code says k, i.
                # Original: Bij[:, :, k, i].T @ Hij_tilde[:, :, k, i]
                # Typo in original code? 
                # Loop variable j is used in Hij_tilde[i, j].
                # Original code: Hij_tilde[:, :, k, i] -> k is child of i.
                # But we are computing for column j.
                # If j is not involved in recursive term, it implies column independence?
                # Let's look at original:
                # Hij_tilde[:, :, i, j] += Bij[:, :, k, i].T @ Hij_tilde[:, :, k, i]
                # Wait, the last index is `i` in original?
                # Let me check the original file provided in context.
                # Line 280: Hij_tilde[:, :, i, j] += Bij[:, :, k, i].T @ Hij_tilde[:, :, k, i]
                # That looks weird. Usually it propagates j.
                # Let's check Hi0_tilde loop.
                # Hi0_tilde[:, :, i] += ... Hij_tilde[:, :, k, i]
                # Maybe Hij_tilde is not (i,j) but (i, something else)?
                # Docs say: Twist-propagation matrix defining the motion coupling between different links.
                # If index j is fixed in outer loop, recursive step should preserve j?
                # Hij_tilde[k, j].
                # If original code had `Hij_tilde[:, :, k, i]`, then it depends on `i`.
                # But `i` is the loop variable. `k` is child of `i`.
                # So `Hij_tilde` for `k` must have been computed.
                # If the second index is `i` (current node), then it's diagonal-ish propagation?
                # But j iterates 0..n.
                
                # Let's assume original code meant `Hij_tilde[:, :, k, j]`.
                # "Hij_tilde[:, :, i, j] = ... + sum(Bij.T @ Hij_tilde[:, :, k, j])"
                # This makes sense for recursive accumulation.
                
                # Re-reading original code from spart_functions.py:
                # 278: Hij_tilde[:, :, i, j] = Mm_tilde[:, :, i] @ Bdotij[:, :, i, j]
                # 280: Hij_tilde[:, :, i, j] += Bij[:, :, k, i].T @ Hij_tilde[:, :, k, i]
                # It explicitly uses `i` as the second index in the recursive call.
                # This means the result for (i,j) depends on result for (k, i).
                # This couples columns? 
                # If so, I must replicate exact logic.
                
                term = Bij[:, :, k, i].T @ Hij_tilde_list[k][i]
                val = val + term
                
            Hij_tilde_list[i][j] = val

    # Hi0_tilde
    Hi0_tilde_list = [None] * n
    for i in reversed(range(n)):
        # Bdot block
        skew_val = skew_symmetric(t0[3:6].flatten() - tL[3:6, i].flatten())
        Bdot = torch.cat([
            torch.cat([zeros, zeros], dim=1),
            torch.cat([skew_val, zeros], dim=1)
        ], dim=0)
        
        val = Mm_tilde[:, :, i] @ Bdot
        children = torch.nonzero(robot['con']['child'][:, i] == 1, as_tuple=True)[0]
        for k in children:
            # Bij[:, :, k, i].T @ Hij_tilde[:, :, k, i]
            term = Bij[:, :, k, i].T @ Hij_tilde_list[k][i]
            val = val + term
        Hi0_tilde_list[i] = val

    # Cm
    Cm_terms = {}
    for j in range(n):
        for i in range(n):
            if robot['joints'][i]['type'] != 0 and robot['joints'][j]['type'] != 0 and (robot['con']['branch'][i, j] == 1 or robot['con']['branch'][j, i] == 1):
                
                qi = robot['joints'][i]['q_id'] - 1
                qj = robot['joints'][j]['q_id'] - 1
                
                if i <= j:
                    children = torch.nonzero(robot['con']['child'][:, j] == 1, as_tuple=True)[0]
                    # sum(...)
                    child_con = torch.zeros((6,6), device=device, dtype=dtype)
                    for k in children:
                        child_con = child_con + Bij[:, :, k, i].T @ Hij_tilde_list[k][j]
                        
                    val = pm[:, i] @ (Bij[:, :, j, i].T @ Mm_tilde[:, :, j] @ Omega_list[j] + child_con + Mdot_tilde[:, :, j]) @ pm[:, j]
                else:
                    val = pm[:, i] @ (Mm_tilde[:, :, i] @ Bij[:, :, i, j] @ Omega_list[j] + Hij_tilde_list[i][j] + Mdot_tilde[:, :, i]) @ pm[:, j]
                
                Cm_terms[(qi, qj)] = val

    rows = []
    for r in range(n_q):
        cols = []
        for c in range(n_q):
            if (r, c) in Cm_terms:
                cols.append(Cm_terms[(r, c)].reshape(1))
            else:
                cols.append(torch.zeros(1, device=device, dtype=dtype))
        rows.append(torch.cat(cols, dim=0))
    Cm = torch.stack(rows, dim=0)

    children = torch.nonzero(robot['con']['child_base'] == 1, as_tuple=True)[0]
    child_con_base = torch.zeros((6,6), device=device, dtype=dtype)
    for k in children:
        child_con_base = child_con_base + Bi0[:, :, k].T @ Hi0_tilde_list[k]
        
    C0 = P0.T @ (M0_tilde @ Omega0 + child_con_base + Mdot0_tilde) @ P0

    # C0m, Cm0
    C0m_cols = []
    # Need q_id mapping
    c0m_dict = {}
    
    for j in range(n):
        if robot['joints'][j]['type'] != 0:
            qj = robot['joints'][j]['q_id'] - 1
            if j == n-1:
                val = P0.T @ (Bi0[:, :, j].T @ Mm_tilde[:, :, j] @ Omega_list[j] + Mdot_tilde[:, :, j]) @ pm[:, j]
            else:
                children = torch.nonzero(robot['con']['child'][:, j] == 1, as_tuple=True)[0]
                child_con = torch.zeros((6,6), device=device, dtype=dtype)
                for k in children:
                    child_con = child_con + Bi0[:, :, k].T @ Hij_tilde_list[k][j]
                
                val = P0.T @ (Bi0[:, :, j].T @ Mm_tilde[:, :, j] @ Omega_list[j] + child_con + Mdot_tilde[:, :, j]) @ pm[:, j]
            c0m_dict[qj] = val

    cols = []
    for k in range(n_q):
        if k in c0m_dict:
            cols.append(c0m_dict[k])
        else:
            cols.append(torch.zeros(6, device=device, dtype=dtype))
    if cols:
        C0m = torch.stack(cols, dim=1)
    else:
        C0m = torch.zeros((6, n_q), device=device, dtype=dtype)

    Cm0_rows = []
    cm0_dict = {}
    for i in range(n):
        if robot['joints'][i]['type'] != 0:
            qi = robot['joints'][i]['q_id'] - 1
            val = pm[:, i] @ (Mm_tilde[:, :, i] @ Bi0[:, :, i] @ Omega0 + Hi0_tilde_list[i] + Mdot_tilde[:, :, i]) @ P0
            cm0_dict[qi] = val
            
    rows = []
    for k in range(n_q):
        if k in cm0_dict:
            rows.append(cm0_dict[k])
        else:
            rows.append(torch.zeros(6, device=device, dtype=dtype))
    if rows:
        Cm0 = torch.stack(rows, dim=0)
    else:
        Cm0 = torch.zeros((n_q, 6), device=device, dtype=dtype)

    return C0, C0m, Cm0, Cm


def jacobian(rp, r0, rL, P0, pm, i, robot):
    # rp -- Position of the point of interest, projected in the inertial CCS -- [3x1]
    # i -- Link id where the point `p` is located -- int 1 to n. 
    device = rp.device
    dtype = rp.dtype
    
    # J0 block
    # J0_mat = torch.eye(6, device=device, dtype=dtype)
    # J0_mat[3:, :3] = skew_symmetric(r0.flatten() - rp.flatten())
    
    zeros = torch.zeros((3, 3), device=device, dtype=dtype)
    I3 = torch.eye(3, device=device, dtype=dtype)
    skew_val = skew_symmetric(r0.flatten() - rp.flatten())
    
    J0_mat = torch.cat([
        torch.cat([I3, zeros], dim=1),
        torch.cat([skew_val, I3], dim=1)
    ], dim=0)
    
    J0 = J0_mat @ P0
    
    # Jm = torch.zeros((6, robot['n_q']), device=device, dtype=dtype)
    # Use dict to collect columns
    jm_dict = {}
    
    for j in range(i):
        if robot['joints'][j]['type'] != 0:
            if robot['con']['branch'][i-1, j] == 1:
                # Jm block
                skew_val_L = skew_symmetric(rL[:, j].flatten() - rp.flatten())
                Jm_block = torch.cat([
                    torch.cat([I3, zeros], dim=1),
                    torch.cat([skew_val_L, I3], dim=1)
                ], dim=0)
                
                val = Jm_block @ pm[:, j]
                jm_dict[robot['joints'][j]['q_id'] - 1] = val
    
    cols = []
    for k in range(robot['n_q']):
        if k in jm_dict:
            cols.append(jm_dict[k])
        else:
            cols.append(torch.zeros(6, device=device, dtype=dtype))
            
    if cols:
        Jm = torch.stack(cols, dim=1)
    else:
        Jm = torch.zeros((6, robot['n_q']), device=device, dtype=dtype)
        
    return J0, Jm
