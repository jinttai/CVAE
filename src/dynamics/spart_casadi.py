import casadi as ca

"""
CasADi‑compatible floating‑base robotics utilities
=================================================
All functions mirror the original NumPy API but operate on **CasADi MX** objects
so the complete dynamics pipeline is symbolically differentiable.  Tensors with
rank > 2 are represented as Python *lists* of 2‑D MX matrices:

* 6 × 6 × n            → ``list[ca.MX(6,6)]`` length *n*
* 6 × 6 × n × n        → ``list[list[ca.MX(6,6)]]`` (row‑major)

Only state variables (pose, velocities, accelerations, joint coordinates) need
be MX; constant robot parameters can stay NumPy/Python.

Example (unchanged call sequence) →

```python
RJ, RL, rJ, rL, e, g               = ft.kinematics(R0, r0, qm, robot)
Bij, Bi0, P0, pm                   = ft.diff_kinematics(R0, r0, rL, e, g, robot)
t0,  tL                            = ft.velocities(Bij, Bi0, P0, pm, u0, um, robot)
t0dot, tLdot                       = ft.accelerations(t0, tL, P0, pm, Bi0, Bij,
                                                     u0, um, u0dot, umdot, robot)
I0, Im                             = ft.inertia_projection(R0, RL, robot)
M0_tilde, Mm_tilde                 = ft.mass_composite_body(I0, Im, Bij, Bi0, robot)
H0, H0m, Hm                        = ft.generalized_inertia_matrix(M0_tilde, Mm_tilde,
                                                                  Bij, Bi0, P0, pm, robot)
C0, C0m, Cm0, Cm                   = ft.convective_inertia_matrix(t0, tL, I0, Im,
                                                                  M0_tilde, Mm_tilde,
                                                                  Bij, Bi0, P0, pm, robot)
```
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zeros(m: int, n: int) -> ca.MX:
    return ca.MX.zeros(m, n)


def _eye(n: int) -> ca.MX:
    return ca.MX.eye(n)


def _skew(v: ca.MX) -> ca.MX:
    v = ca.reshape(v, 3, 1)
    z = ca.MX(0)
    return ca.vertcat(
        ca.hcat([   z, -v[2],  v[1]]),
        ca.hcat([ v[2],    z, -v[0]]),
        ca.hcat([-v[1],  v[0],    z])
    )


def _bmat(rows) -> ca.MX:
    return ca.vertcat(*[ca.hcat(r) for r in rows])

# ---------------------------------------------------------------------------
# Orientation utilities
# ---------------------------------------------------------------------------

def euler_dcm(e: ca.MX, alpha: ca.MX) -> ca.MX:
    q_vec = e * ca.sin(alpha / 2)
    q0    = ca.cos(alpha / 2)
    return quat_dcm(ca.vertcat(q_vec, q0))


def quat_dcm(q: ca.MX) -> ca.MX:
    q1, q2, q3, q0 = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.hcat([1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)]),
        ca.hcat([2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)]),
        ca.hcat([2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)])
    )
    
def quat_dot(q: ca.MX, w: ca.MX) -> ca.MX:
    # w = [w1, w2, w3] is angular velocity in body-fixed CCS
    q1, q2, q3, q0 = q[0], q[1], q[2], q[3]
    w1, w2, w3     = w[0], w[1], w[2]
    q_dot = ca.vertcat(
        0.5 * ( q0 * w1 + q2 * w3 - q3 * w2),
        0.5 * ( q0 * w2 + q3 * w1 - q1 * w3),
        0.5 * ( q0 * w3 + q1 * w2 - q2 * w1),
        0.5 * (-q1 * w1 - q2 * w2 - q3 * w3)
        
    )
    return q_dot

def dcm_quat(DCM: ca.MX) -> ca.MX:
    
    r11 = DCM[0, 0]
    r12 = DCM[0, 1]
    r13 = DCM[0, 2]
    r21 = DCM[1, 0]
    r22 = DCM[1, 1]
    r23 = DCM[1, 2]
    r31 = DCM[2, 0]
    r32 = DCM[2, 1]
    r33 = DCM[2, 2]
    
    trace = r11 + r22 + r33
    
    w1 = 0.5 * ca.sqrt(1 + trace)
    x1 = (r32 - r23) / (4 * w1)
    y1 = (r13 - r31) / (4 * w1)
    z1 = (r21 - r12) / (4 * w1)
    
    x2 = 0.5 * ca.sqrt(1 + r11 - r22 - r33)
    w2 = (r32 - r23) / (4 * x2)
    y2 = (r12 + r21) / (4 * x2)
    z2 = (r13 + r31) / (4 * x2)
    
    y3 = 0.5 * ca.sqrt(1 - r11 + r22 - r33)
    w3 = (r13 - r31) / (4 * y3)
    x3 = (r12 + r21) / (4 * y3)
    z3 = (r23 + r32) / (4 * y3)
    
    z4 = 0.5 * ca.sqrt(1 - r11 - r22 + r33)
    w4 = (r21 - r12) / (4 * z4)
    x4 = (r13 + r31) / (4 * z4)
    y4 = (r23 + r32) / (4 * z4)
    
    quat = ca.MX.zeros(4)
    
    cond1 = trace > 0
    cond2 = ca.logic_and(ca.logic_not(cond1), ca.logic_and(r11 > r22, r11 > r33))
    cond3 = ca.logic_and(ca.logic_not(ca.logic_or(cond1, cond2)), r22 > r33)
    
    # x 
    quat[0] = ca.if_else(cond1, x1, 
                ca.if_else(cond2, x2,
                    ca.if_else(cond3, x3, x4)))
    # y 
    quat[1] = ca.if_else(cond1, y1, 
                ca.if_else(cond2, y2,
                    ca.if_else(cond3, y3, y4)))
    # z 
    quat[2] = ca.if_else(cond1, z1, 
                ca.if_else(cond2, z2,
                    ca.if_else(cond3, z3, z4)))
    # w
    quat[3] = ca.if_else(cond1, w1, 
                ca.if_else(cond2, w2,
                    ca.if_else(cond3, w3, w4)))
    return quat

# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def kinematics(R0: ca.MX, r0: ca.MX, qm: ca.MX, robot: dict):
    n = robot['n_links_joints']
    T0 = _bmat([[R0, r0], [ca.MX.zeros(1, 3), ca.MX(1)]])
    TJ, TL = [None]*n, [None]*n
    for i in range(n):
        joint   = robot['joints'][i]
        parentL = joint['parent_link']
        TJ[i]   = (T0 if parentL == 0 else TL[parentL - 1]) @ joint['T']
        jtype = joint['type']
        qi    = qm[joint['q_id'] - 1] if jtype else ca.MX(0)
        axis  = joint['axis']
        if jtype == 1:
            T_q = _bmat([[euler_dcm(axis, qi), _zeros(3, 1)], [ca.MX.zeros(1,3), ca.MX(1)]])
        elif jtype == 2:
            T_q = _bmat([[ _eye(3), ca.reshape(axis*qi, 3, 1)], [ca.MX.zeros(1,3), ca.MX(1)]])
        else:
            T_q = ca.MX.eye(4)
        link  = robot['links'][joint['child_link'] - 1]
        TL[i] = TJ[i] @ T_q @ link['T']
    RJ = ca.hcat([ca.reshape(TJ[i][0:3,0:3], 9, 1) for i in range(n)])
    RL = ca.hcat([ca.reshape(TL[i][0:3,0:3], 9, 1) for i in range(n)])
    rJ = ca.hcat([TJ[i][0:3,3] for i in range(n)])
    rL = ca.hcat([TL[i][0:3,3] for i in range(n)])
    e  = ca.hcat([ca.mtimes(ca.reshape(RJ[:,i],3,3), robot['joints'][i]['axis']) for i in range(n)])
    g  = ca.hcat([rL[:,i] - rJ[:, robot['links'][i]['parent_joint'] - 1] for i in range(n)])
    return RJ, RL, rJ, rL, e, g
# ---------------------------------------------------------------------------
# Differential kinematics building blocks
# ---------------------------------------------------------------------------

def diff_kinematics(R0: ca.MX, r0: ca.MX, rL: ca.MX, e: ca.MX, g: ca.MX, robot: dict):
    n = robot['n_links_joints']

    Bij = [[_zeros(6,6) for _ in range(n)] for _ in range(n)]
    Bi0 = [_zeros(6,6) for _ in range(n)]
    pm  = _zeros(6, n)

    P0 = _bmat([[R0, _zeros(3,3)], [_zeros(3,3), _eye(3)]])

    for i in range(n):
        rL_i = rL[:,i]
        for j in range(n):
            if robot['con']['branch'][i][j] == 1:
                Bij[i][j] = _bmat([[ _eye(3),                _zeros(3,3)],
                                   [_skew(rL[:,j] - rL_i),  _eye(3)     ]])
        Bi0[i] = _bmat([[ _eye(3),               _zeros(3,3)],
                        [_skew(r0 - rL_i),       _eye(3)    ]])

        jtype = robot['joints'][i]['type']
        if jtype == 1:   # revolute
            pm[:,i] = ca.vertcat(e[:,i], ca.cross(e[:,i], g[:,i]))
        elif jtype == 2: # prismatic
            pm[:,i] = ca.vertcat(_zeros(3,1), e[:,i])
        else:
            pm[:,i] = _zeros(6,1)

    return Bij, Bi0, P0, pm

# ---------------------------------------------------------------------------
# Link and base twists
# ---------------------------------------------------------------------------

def velocities(Bij, Bi0, P0, pm, u0, um, robot):
    n  = robot['n_links_joints']
    t0 = P0 @ u0
    tL_tmp = [None]*n

    for i in range(n):
        parent = robot['joints'][i]['parent_link']
        tLi = Bi0[i] @ t0 if parent == 0 else Bij[i][parent-1] @ tL_tmp[parent-1]
        if robot['joints'][i]['type'] != 0:
            qi = robot['joints'][i]['q_id'] - 1
            tLi += pm[:,i] * um[qi]
        tL_tmp[i] = tLi

    tL = ca.hcat(tL_tmp)
    return t0, tL

# ---------------------------------------------------------------------------
# Accelerations
# ---------------------------------------------------------------------------

def accelerations(t0, tL, P0, pm, Bi0, Bij, u0, um, u0dot, umdot, robot):
    n = robot['n_links_joints']

    Omega0 = _bmat([[ _skew(t0[0:3]), _zeros(3,3)], [_zeros(3,3), _zeros(3,3)]])
    Omegam = [_bmat([[ _skew(tL[0:3,i]), _zeros(3,3)], [_zeros(3,3), _skew(tL[0:3,i])]]) for i in range(n)]

    t0dot = Omega0 @ P0 @ u0 + P0 @ u0dot

    tLdot_tmp = [None]*n
    for i in range(n):
        parent = robot['joints'][i]['parent_link']
        if parent == 0:
            skew_diff = _skew(t0[3:6] - tL[3:6,i])
            aux = _bmat([[ _zeros(3,6)], [skew_diff, _zeros(3,3)]])
            tLi_dot = Bi0[i] @ t0dot + aux @ t0
        else:
            p_idx = parent-1
            skew_diff = _skew(tL[3:6,p_idx] - tL[3:6,i])
            aux = _bmat([[ _zeros(3,6)], [skew_diff, _zeros(3,3)]])
            tLi_dot = Bij[i][p_idx] @ tLdot_tmp[p_idx] + aux @ tL[:,p_idx]
        if robot['joints'][i]['type'] != 0:
            qi = robot['joints'][i]['q_id'] - 1
            tLi_dot += Omegam[i] @ pm[:,i] * um[qi] + pm[:,i] * umdot[qi]
        tLdot_tmp[i] = tLi_dot

    tLdot = ca.hcat(tLdot_tmp)
    return t0dot, tLdot

# ---------------------------------------------------------------------------
# Centre of mass of whole system
# ---------------------------------------------------------------------------

def center_of_mass(r0: ca.MX, rL: ca.MX, robot: dict):
    m_total = robot['base_link']['mass']
    mass_r  = r0 * m_total
    for i in range(robot['n_links_joints']):
        mi      = robot['links'][i]['mass']
        m_total += mi
        mass_r += rL[:,i] * mi
    return mass_r / m_total

# ---------------------------------------------------------------------------
# Inertia projection (spatial 6×6) — uses RL flat matrix
# ---------------------------------------------------------------------------

def inertia_projection(R0: ca.MX, RL: ca.MX, robot: dict):
    I0 = R0 @ robot['base_link']['inertia'] @ R0.T
    n  = robot['n_links_joints']
    Im = [None]*n
    for i in range(n):
        Ri = ca.reshape(RL[:,i], (3,3))
        Im[i] = Ri @ robot['links'][i]['inertia'] @ Ri.T
    return I0, Im

# ---------------------------------------------------------------------------
# Composite rigid‑body inertia (CRBA sweep)
# ---------------------------------------------------------------------------

def mass_composite_body(I0, Im, Bij, Bi0, robot):
    n = robot['n_links_joints']
    Mm_tilde = [None]*n
    for i in reversed(range(n)):
        Mi = _bmat([[Im[i], _zeros(3,3)], [_zeros(3,3), robot['links'][i]['mass'] * _eye(3)]])
        children = [k for k in range(n) if robot['con']['child'][k][i] == 1]
        for j in children:
            Mi += Bij[j][i].T @ Mm_tilde[j] @ Bij[j][i]
        Mm_tilde[i] = Mi

    M0_tilde = _bmat([[I0, _zeros(3,3)], [_zeros(3,3), robot['base_link']['mass'] * _eye(3)]])
    for j in [k for k in range(n) if robot['con']['child_base'][k] == 1]:
        M0_tilde += Bi0[j].T @ Mm_tilde[j] @ Bi0[j]

    return M0_tilde, Mm_tilde

# ---------------------------------------------------------------------------
# Generalised inertia matrix
# ---------------------------------------------------------------------------

def generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot):
    n_q = robot['n_q']
    n   = robot['n_links_joints']
    H0  = P0.T @ M0_tilde @ P0
    Hm  = _zeros(n_q, n_q)
    H0m = _zeros(6, n_q)

    for j in range(n):
        if robot['joints'][j]['type'] == 0: continue
        qj = robot['joints'][j]['q_id'] - 1
        for i in range(j, n):
            if robot['joints'][i]['type'] == 0: continue
            qi = robot['joints'][i]['q_id'] - 1
            val = pm[:,i].T @ Mm_tilde[i] @ Bij[i][j] @ pm[:,j]
            Hm[qi,qj] = val
            Hm[qj,qi] = val
        H0m[:,qj] = (pm[:,j].T @ Mm_tilde[j] @ Bi0[j] @ P0).T
    return H0, H0m, Hm

# ---------------------------------------------------------------------------
# Convective inertia matrix (Coriolis/centrifugal)
# ---------------------------------------------------------------------------

def convective_inertia_matrix(t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot):
    n_q = robot['n_q']; n = robot['n_links_joints']; Z3 = _zeros(3,3)
    def big_omega(w): return _bmat([[ _skew(w[0:3]), Z3], [Z3, Z3]])
    def _bmat2(w): return _bmat([[ _skew(w[0:3]), Z3], [Z3, _skew(w[0:3])]])
    Omega0 = big_omega(t0); Omega = [_bmat2(tL[:,i]) for i in range(n)]
    Mdot0 = _bmat([[Omega0[0:3,0:3] @ I0, Z3], [Z3, Z3]])
    Mdot  = [_bmat([[Omega[i][0:3,0:3] @ Im[i], Z3], [Z3, Z3]]) for i in range(n)]
    Mdot_tilde = [None]*n
    for i in reversed(range(n)):
        Md = Mdot[i]
        for j in [k for k in range(n) if robot['con']['child'][k][i]]: Md += Mdot_tilde[j]
        Mdot_tilde[i] = Md
    Mdot0_tilde = Mdot0
    base_children = [k for k in range(n) if robot['con']['child_base'][k]]
    for k in base_children: Mdot0_tilde += Mdot_tilde[k]
    Bdotij = [[_zeros(6,6) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if robot['con']['branch'][i][j]:
                Bdotij[i][j] = _bmat([[Z3, Z3], [_skew(tL[3:6,j]-tL[3:6,i]), Z3]])
    Hij_tilde = [[_zeros(6,6) for _ in range(n)] for _ in range(n)]
    for i in reversed(range(n)):
        for j in reversed(range(n)):
            H = Mm_tilde[i] @ Bdotij[i][j]
            for k in [c for c in range(n) if robot['con']['child'][c][i]]:
                H += Bij[k][i].T @ Hij_tilde[k][i]
            Hij_tilde[i][j] = H
    Hi0_tilde = [_zeros(6,6) for _ in range(n)]
    for i in reversed(range(n)):
        Bdot = _bmat([[Z3, Z3], [_skew(t0[3:6]-tL[3:6,i]), Z3]])
        H = Mm_tilde[i] @ Bdot
        for k in [c for c in range(n) if robot['con']['child'][c][i]]:
            H += Bij[k][i].T @ Hij_tilde[k][i]
        Hi0_tilde[i] = H
    Cm = _zeros(n_q, n_q)
    for i in range(n):
        if robot['joints'][i]['type'] == 0: continue
        qi = robot['joints'][i]['q_id'] - 1
        for j in range(n):
            if robot['joints'][j]['type'] == 0: continue
            qj = robot['joints'][j]['q_id'] - 1
            if robot['con']['branch'][i][j] or robot['con']['branch'][j][i]:
                if i <= j:
                    child_term = sum(Bij[k][i].T @ Hij_tilde[k][j] for k in [c for c in range(n) if robot['con']['child'][c][j]])
                    term = Bij[j][i].T @ Mm_tilde[j] @ Omega[j] + child_term + Mdot_tilde[j]
                else:
                    term = Mm_tilde[i] @ Bij[i][j] @ Omega[j] + Hij_tilde[i][j] + Mdot_tilde[i]
                Cm[qi,qj] = pm[:,i].T @ term @ pm[:,j]
    child_term0 = sum(Bi0[k].T @ Hi0_tilde[k] for k in base_children)
    C0 = P0.T @ (M0_tilde @ Omega0 + child_term0 + Mdot0_tilde) @ P0
    C0m = _zeros(6, n_q); Cm0 = _zeros(n_q, 6)
    for j in range(n):
        if robot['joints'][j]['type'] == 0: continue
        qj = robot['joints'][j]['q_id'] - 1
        child_term = sum(Bi0[k].T @ Hij_tilde[k][j] for k in [c for c in range(n) if robot['con']['child'][c][j]])
        term_j = Bi0[j].T @ Mm_tilde[j] @ Omega[j] + child_term + Mdot_tilde[j]
        C0m[:,qj] = ca.reshape(P0.T @ term_j @ pm[:,j], 6, 1)
        Cm0[qj,:] = ca.reshape(pm[:,j].T @ (Mm_tilde[j] @ Bi0[j] @ Omega0 + Hi0_tilde[j] + Mdot_tilde[j]) @ P0, 1, 6)
    return C0, C0m, Cm0, Cm

# ---------------------------------------------------------------------------
# Geometric Jacobian
# ---------------------------------------------------------------------------

def jacobian(rp: ca.MX, r0: ca.MX, rL: ca.MX, P0: ca.MX, pm: ca.MX, link_id: int, robot: dict):
    """Jacobian of point *rp* fixed on link *link_id* (1‑based)."""
    J0 = _bmat([[ _eye(3), _zeros(3,3)], [_skew(r0 - rp), _eye(3)]]) @ P0
    n_q = robot['n_q']
    Jm  = _zeros(6, n_q)
    for j in range(link_id):
        if robot['joints'][j]['type'] == 0: continue
        if robot['con']['branch'][link_id-1][j]:
            qj = robot['joints'][j]['q_id'] - 1
            Jm[:,qj] = _bmat([[ _eye(3), _zeros(3,3)], [_skew(rL[:,j] - rp), _eye(3)]]) @ pm[:,j]
    return J0, Jm
