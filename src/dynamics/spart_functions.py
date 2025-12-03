import numpy as np

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
    v = np.asarray(v, dtype=np.float32).flatten()
    assert len(v) == 3, 'vector length error'
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float32)

def euler_dcm(e, alpha):
    e = np.asarray(e, dtype=np.float32)
    alpha = np.float32(alpha)
    q = np.hstack((e * np.sin(alpha / 2), np.cos(alpha / 2)))
    return quat_dcm(q)

def quat_dcm(q):
    q = np.asarray(q, dtype=np.float32).flatten()
    assert len(q) == 4, 'quaternion length error'
    q1, q2, q3, q0 = q
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)]
    ], dtype=np.float32)

def quat_dot(q, w):
    q = np.asarray(q, dtype=np.float32).flatten()
    w = np.asarray(w, dtype=np.float32).flatten()
    assert len(q) == 4 and len(w) == 3, 'quaternion or angular velocity length error'
    q1, q2, q3, q0 = q
    w1, w2, w3 = w
    return np.array([
        -0.5 * (w1 * q2 + w2 * q3 + w3 * q0),
        0.5 * (w1 * q0 - w2 * q3 + w3 * q2),
        0.5 * (w2 * q0 + w1 * q3 - w3 * q1),
        0.5 * (w3 * q0 - w1 * q2 + w2 * q1)
    ], dtype=np.float32)

def accelerations(t0, tL, P0, pm, Bi0, Bij, u0, um, u0dot, umdot, robot):
    n = robot['n_links_joints']
    Omega0 = np.block([[skew_symmetric(t0[:3]), np.zeros((3,3), dtype=np.float32)],
                       [np.zeros((3,3), dtype=np.float32), np.zeros((3,3), dtype=np.float32)]])
    Omegam = np.zeros((6,6,n), dtype=np.float32)
    for i in range(n):
        skew_tLi = skew_symmetric(tL[:3, i])
        Omegam[:, :, i] = np.block([[skew_tLi, np.zeros((3,3), dtype=np.float32)],
                                    [np.zeros((3,3), dtype=np.float32), skew_tLi]])
    t0dot = Omega0 @ P0 @ u0 + P0 @ u0dot
    tLdot = np.zeros((6, n), dtype=np.float32)
    for i in range(n):
        parent_link = robot['joints'][i]['parent_link']
        if parent_link == 0:
            skew_diff = skew_symmetric(t0[3:6].flatten() - tL[3:6, i])
            tLdot[:, i] = (Bi0[:, :, i] @ t0dot + np.block([[np.zeros((3,6), dtype=np.float32)], 
                                                            [skew_diff, np.zeros((3,3), dtype=np.float32)]]) @ t0).flatten()
        else:
            skew_diff = skew_symmetric(tL[3:6, parent_link-1] - tL[3:6, i])
            tLdot[:, i] = (Bij[:, :, i, parent_link-1] @ tLdot[:, parent_link-1] + 
                           np.block([[np.zeros((3,6), dtype=np.float32)], 
                                     [skew_diff, np.zeros((3,3), dtype=np.float32)]]) @ tL[:, parent_link-1]).flatten()
        if robot['joints'][i]['type'] != 0:
            q_id = robot['joints'][i]['q_id']
            tLdot[:, i] += Omegam[:, :, i] @ pm[:, i] * um[q_id-1] + pm[:, i] * umdot[q_id-1]
    return t0dot, tLdot

def center_of_mass(r0, rL, robot):
    mass_total = np.float32(robot['base_link']['mass'])
    mass_r = r0 * robot['base_link']['mass']
    for i in range(robot['n_links_joints']):
        mass_total += robot['links'][i]['mass']
        mass_r += rL[:, i].reshape(3,1) * robot['links'][i]['mass']
    return mass_r / mass_total

def kinematics(R0, r0, qm, robot):
    n = robot['n_links_joints']
    T0 = np.block([[R0, r0.reshape(3,1)], [np.zeros((1,3), dtype=np.float32), 1]])
    TJ = np.zeros((4, 4, n), dtype=np.float32)
    TL = np.zeros((4, 4, n), dtype=np.float32)
    for i in range(n):
        joint = robot['joints'][i]
        if joint['parent_link'] == 0:
            TJ[:, :, joint['id'] - 1] = T0 @ joint['T']
        else:
            TJ[:, :, joint['id'] - 1] = TL[:, :, joint['parent_link'] - 1] @ joint['T']
        if joint['type'] == 1:
            T_qm = np.block([[euler_dcm(joint['axis'], qm[joint['q_id'] - 1]), np.zeros((3,1), dtype=np.float32)],
                             [np.zeros((1,3), dtype=np.float32), 1]])
        elif joint['type'] == 2:
            T_qm = np.block([[np.eye(3, dtype=np.float32), joint['axis'].reshape(3,1) * qm[joint['q_id'] - 1]],
                             [np.zeros((1,3), dtype=np.float32), 1]])
        else:
            T_qm = np.eye(4, dtype=np.float32)
        link = robot['links'][joint['child_link'] - 1]
        TL[:, :, link['id'] - 1] = TJ[:, :, link['parent_joint'] - 1] @ T_qm @ link['T']
    RJ = TJ[:3, :3, :]
    RL = TL[:3, :3, :]
    rJ = TJ[:3, 3, :]
    rL = TL[:3, 3, :]
    e = np.zeros((3, n), dtype=np.float32)
    g = np.zeros((3, n), dtype=np.float32)
    for i in range(n):
        e[:, i] = RJ[:, :, i] @ robot['joints'][i]['axis']
        g[:, i] = rL[:, i] - rJ[:, robot['links'][i]['parent_joint'] - 1]
    return RJ, RL, rJ, rL, e, g

def diff_kinematics(R0, r0, rL, e, g, robot):
    n = robot['n_links_joints']
    Bij = np.zeros((6, 6, n, n), dtype=np.float32)
    Bi0 = np.zeros((6, 6, n), dtype=np.float32)
    pm = np.zeros((6, n), dtype=np.float32)
    P0 = np.block([[R0, np.zeros((3,3), dtype=np.float32)], [np.zeros((3,3), dtype=np.float32), np.eye(3, dtype=np.float32)]])
    for i in range(n):
        for j in range(n):
            if robot['con']['branch'][i, j] == 1:
                Bij[:, :, i, j] = np.block([[np.eye(3, dtype=np.float32), np.zeros((3,3), dtype=np.float32)],
                                            [skew_symmetric(rL[:, j] - rL[:, i]), np.eye(3, dtype=np.float32)]])
        Bi0[:, :, i] = np.block([[np.eye(3, dtype=np.float32), np.zeros((3,3), dtype=np.float32)],
                                    [skew_symmetric(r0 - rL[:, i].reshape(3,1)), np.eye(3, dtype=np.float32)]])
        if robot['joints'][i]['type'] == 1:
            pm[:, i] = np.hstack((e[:, i], np.cross(e[:, i], g[:, i])))
        elif robot['joints'][i]['type'] == 2:
            pm[:, i] = np.hstack((np.zeros(3, dtype=np.float32), e[:, i]))
        else:
            pm[:, i] = np.zeros(6, dtype=np.float32)
    return Bij, Bi0, P0, pm

def velocities(Bij, Bi0, P0, pm, u0, um, robot):
    n = robot['n_links_joints']
    t0 = P0 @ u0
    tL = np.zeros((6, n), dtype=np.float32)
    for i in range(n):
        parent_link = robot['joints'][i]['parent_link']
        if parent_link == 0:
            tL[:, i] = (Bi0[:, :, i] @ t0).flatten()
        else:
            tL[:, i] = Bij[:, :, i, i - 1] @ tL[:, i - 1]
        if robot['joints'][i]['type'] != 0:
            q_id = robot['joints'][i]['q_id']
            tL[:, i] += pm[:, i] * um[q_id-1]
    return t0, tL

def inertia_projection(R0, RL, robot):
    I0 = R0 @ robot['base_link']['inertia'] @ R0.T
    n = robot['n_links_joints']
    Im = np.zeros((3, 3, n), dtype=np.float32)
    for i in range(n):
        Im[:, :, i] = RL[:, :, i] @ robot['links'][i]['inertia'] @ RL[:, :, i].T
    return I0, Im

def mass_composite_body(I0, Im, Bij, Bi0, robot):
    n = robot['n_links_joints']
    Mm_tilde = np.zeros((6, 6, n), dtype=np.float32)
    for i in reversed(range(n)):
        Mm_tilde[:, :, i] = np.block([
            [Im[:, :, i], np.zeros((3, 3), dtype=np.float32)],
            [np.zeros((3, 3), dtype=np.float32), robot['links'][i]['mass'] * np.eye(3, dtype=np.float32)]
        ])
        children = np.where(robot['con']['child'][:, i] == 1)[0]
        for j in children:
            Mm_tilde[:, :, i] += Bij[:, :, j, i].T @ Mm_tilde[:, :, j] @ Bij[:, :, j, i]
    M0_tilde = np.block([
        [I0, np.zeros((3, 3), dtype=np.float32)],
        [np.zeros((3, 3), dtype=np.float32), robot['base_link']['mass'] * np.eye(3, dtype=np.float32)]
    ])
    children = np.where(robot['con']['child_base'] == 1)[0]
    for j in children:
        M0_tilde += Bi0[:, :, j].T @ Mm_tilde[:, :, j] @ Bi0[:, :, j]
    return M0_tilde, Mm_tilde

def generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot):
    n_q = robot['n_q']
    n = robot['n_links_joints']
    H0 = P0.T @ M0_tilde @ P0
    Hm = np.zeros((n_q, n_q), dtype=np.float32)
    for j in range(n):
        for i in range(j, n):
            if robot['joints'][i]['type'] != 0 and robot['joints'][j]['type'] != 0:
                qi = robot['joints'][i]['q_id'] - 1
                qj = robot['joints'][j]['q_id'] - 1
                if qi >= 0 and qj >= 0:
                    val = (pm[:6, i].T 
                            @ Mm_tilde[:6, :6, i] 
                            @ Bij[:6, :6, i, j] 
                            @ pm[:6, j])
                    Hm[qi, qj] = val
                    Hm[qj, qi] = val  
    H0m = np.zeros((6, n_q), dtype=np.float32)
    for i in range(n):
        if robot['joints'][i]['type'] != 0:
            qi = robot['joints'][i]['q_id'] - 1
            vec = pm[:6, i].T @ Mm_tilde[:6, :6, i] @ Bi0[:6, :6, i] @ P0
            H0m[:, qi] = vec.T
    return H0, H0m, Hm

def convective_inertia_matrix(t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot):
    n_q = robot['n_q']
    n = robot['n_links_joints']
    Omega0 = np.block([[skew_symmetric(t0[:3]), np.zeros((3,3), dtype=np.float32)], [np.zeros((3,3), dtype=np.float32), np.zeros((3,3), dtype=np.float32)]])
    Omega = np.zeros((6, 6, n), dtype=np.float32)
    for i in range(n):
        Omega[:, :, i] = np.block([[skew_symmetric(tL[:3, i]), np.zeros((3,3), dtype=np.float32)],
                                    [np.zeros((3,3), dtype=np.float32), skew_symmetric(tL[:3, i])]])
        
    Mdot0 = np.block([[Omega0[:3,:3] @ I0, np.zeros((3,3), dtype=np.float32)], [np.zeros((3,3), dtype=np.float32), np.zeros((3,3), dtype=np.float32)]])
    Mdot = np.zeros((6, 6, n), dtype=np.float32)
    for i in range(n):
        Mdot[:, :, i] = np.block([[Omega[:3, :3, i] @ Im[:, :, i], np.zeros((3,3), dtype=np.float32)],
                                    [np.zeros((3,3), dtype=np.float32), np.zeros((3,3), dtype=np.float32)]])

    Mdot_tilde = np.zeros((6, 6, n), dtype=np.float32)
    for i in reversed(range(n)):
        Mdot_tilde[:, :, i] = Mdot[:, :, i]
        children = np.where(robot['con']['child'][:, i] == 1)[0]
        for j in children:
            Mdot_tilde[:, :, i] += Mdot_tilde[:, :, j]

    Mdot0_tilde = Mdot0
    children = np.where(robot['con']['child_base'] == 1)[0]
    for j in children:
        Mdot0_tilde += Mdot_tilde[:, :, j]

    Bdotij = np.zeros((6, 6, n, n), dtype=np.float32)
    for j in range(n):
        for i in range(n):
            if robot['con']['branch'][i, j] == 1:
                Bdotij[:, :, i, j] = np.block([[np.zeros((3,3), dtype=np.float32), np.zeros((3,3), dtype=np.float32)],
                                                [skew_symmetric(tL[3:6, j] - tL[3:6, i]), np.zeros((3,3), dtype=np.float32)]])

    Hij_tilde = np.zeros((6, 6, n, n), dtype=np.float32)
    for i in reversed(range(n)):
        for j in reversed(range(n)):
            Hij_tilde[:, :, i, j] = Mm_tilde[:, :, i] @ Bdotij[:, :, i, j]
            children = np.where(robot['con']['child'][:, i] == 1)[0]
            for k in children:
                Hij_tilde[:, :, i, j] += Bij[:, :, k, i].T @ Hij_tilde[:, :, k, i]

    Hi0_tilde = np.zeros((6, 6, n), dtype=np.float32)
    for i in reversed(range(n)):
        Bdot = np.block([[np.zeros((3,3), dtype=np.float32), np.zeros((3,3), dtype=np.float32)],
                            [skew_symmetric(t0[3:6] - tL[3:6, i].reshape(3,1)), np.zeros((3,3), dtype=np.float32)]])
        Hi0_tilde[:, :, i] = Mm_tilde[:, :, i] @ Bdot
        children = np.where(robot['con']['child'][:, i] == 1)[0]
        for k in children:
            Hi0_tilde[:, :, i] += Bij[:, :, k, i].T @ Hij_tilde[:, :, k, i]

    Cm = np.zeros((n_q, n_q), dtype=np.float32)
    for j in range(n):
        for i in range(n):
            if robot['joints'][i]['type'] != 0 and robot['joints'][j]['type'] != 0 and (robot['con']['branch'][i, j] == 1 or robot['con']['branch'][j, i] == 1):
                if i <= j:
                    children = np.where(robot['con']['child'][:, j] == 1)[0]
                    child_con = sum(Bij[:, :, k, i].T @ Hij_tilde[:, :, k, j] for k in children)
                    Cm[robot['joints'][i]['q_id'] - 1, robot['joints'][j]['q_id'] - 1] = (
                        pm[:, i].T @ (Bij[:, :, j, i].T @ Mm_tilde[:, :, j] @ Omega[:, :, j] + child_con + Mdot_tilde[:, :, j]) @ pm[:, j]
                    )
                else:
                    Cm[robot['joints'][i]['q_id'] - 1, robot['joints'][j]['q_id'] - 1] = (
                        pm[:, i].T @ (Mm_tilde[:, :, i] @ Bij[:, :, i, j] @ Omega[:, :, j] + Hij_tilde[:, :, i, j] + Mdot_tilde[:, :, i]) @ pm[:, j]
                    )

    children = np.where(robot['con']['child_base'] == 1)[0]
    child_con = sum(Bi0[:, :, k].T @ Hi0_tilde[:, :, k] for k in children)
    C0 = P0.T @ (M0_tilde @ Omega0 + child_con + Mdot0_tilde) @ P0

    C0m = np.zeros((6, n_q), dtype=np.float32)
    for j in range(n):
        if robot['joints'][j]['type'] != 0:
            if j == n-1:
                C0m[:, robot['joints'][j]['q_id'] - 1] = P0.T @ (Bi0[:, :, j].T @ Mm_tilde[:, :, j] @ Omega[:, :, j] + Mdot_tilde[:, :, j]) @ pm[:, j]
            else:
                children = np.where(robot['con']['child'][:, j] == 1)[0]
                child_con = sum(Bi0[:, :, k].T @ Hij_tilde[:, :, k, j] for k in children)
                C0m[:, robot['joints'][j]['q_id'] - 1] = P0.T @ (Bi0[:, :, j].T @ Mm_tilde[:, :, j] @ Omega[:, :, j] + child_con + Mdot_tilde[:, :, j]) @ pm[:, j]

    Cm0 = np.zeros((n_q, 6), dtype=np.float32)
    for i in range(n):
        if robot['joints'][i]['type'] != 0:
            Cm0[robot['joints'][i]['q_id'] - 1, :] = pm[:, i].T @ (Mm_tilde[:, :, i] @ Bi0[:, :, i] @ Omega0 + Hi0_tilde[:, :, i] + Mdot_tilde[:, :, i]) @ P0

    return C0, C0m, Cm0, Cm


def jacobian(rp, r0, rL, P0, pm, i, robot):
    # rp -- Position of the point of interest, projected in the inertial CCS -- [3x1]
    # i -- Link id where the point `p` is located -- int 1 to n. 
    J0 = np.block([[np.eye(3, dtype=np.float32), np.zeros((3,3), dtype=np.float32)], [skew_symmetric(r0.reshape(3,1) - rp.reshape(3,1)), np.eye(3, dtype=np.float32)]]) @ P0
    Jm = np.zeros((6, robot['n_q']), dtype=np.float32)
    for j in range(i):
        if robot['joints'][j]['type'] != 0:
            if robot['con']['branch'][i-1, j] == 1:
                Jm[:, robot['joints'][j]['q_id'] - 1] = np.block([[np.eye(3, dtype=np.float32), np.zeros((3,3), dtype=np.float32)],
                                                                    [skew_symmetric(rL[:, j].reshape(3,1)  - rp.reshape(3,1) ), np.eye(3, dtype=np.float32)]]) @ pm[:, j]
    return J0, Jm
