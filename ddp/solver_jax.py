"""
DDP/iLQR Solver
JAX implementation of Differential Dynamic Programming / Iterative LQR.
"""

import jax
import jax.numpy as jnp
from jax import grad, jacrev, hessian
import ddp.dynamics_jax as dynamics
import ddp.cost_jax as cost


def compute_dynamics_jacobians(dynamics_model, state, control, dt):
    """
    Compute dynamics Jacobians: f_x, f_u
    
    Args:
        dynamics_model: SpaceRobotDynamics instance
        state: [10] current state
        control: [6] current control
        dt: time step
    
    Returns:
        f_x: [10, 10] Jacobian w.r.t. state
        f_u: [10, 6] Jacobian w.r.t. control
    """
    def dynamics_fn(s, u):
        return dynamics_model.step(s, u, dt)
    
    # Compute Jacobians using JAX
    f_x = jacrev(lambda s: dynamics_fn(s, control))(state)
    f_u = jacrev(lambda u: dynamics_fn(state, u))(control)
    
    return f_x, f_u


def compute_dynamics_hessians(dynamics_model, state, control, dt):
    """
    Compute dynamics Hessians: f_xx, f_uu, f_ux (for full DDP curvature terms)
    
    Args:
        dynamics_model: SpaceRobotDynamics instance
        state: [10] current state
        control: [6] current control
        dt: time step
    
    Returns:
        f_xx: [10, 10, 10] Hessian tensor w.r.t. state (f_xx[i] is Hessian of f[i])
        f_uu: [10, 6, 6] Hessian tensor w.r.t. control (f_uu[i] is Hessian of f[i])
        f_ux: [10, 6, 10] Mixed Hessian tensor (f_ux[i] is mixed Hessian of f[i])
    """
    def dynamics_fn(s, u):
        return dynamics_model.step(s, u, dt)
    
    n_x = state.shape[0]  # 10
    n_u = control.shape[0]  # 6
    
    # Initialize Hessian tensors
    f_xx = jnp.zeros((n_x, n_x, n_x))
    f_uu = jnp.zeros((n_x, n_u, n_u))
    f_ux = jnp.zeros((n_x, n_u, n_x))
    
    # Compute Hessian for each output dimension
    for i in range(n_x):
        # f_xx[i]: Hessian of f[i] w.r.t. state
        def fi_x(s):
            return dynamics_fn(s, control)[i]
        
        # Compute Hessian using JAX
        f_xx_i = hessian(fi_x)(state)
        f_xx = f_xx.at[i].set(f_xx_i)
        
        # f_uu[i]: Hessian of f[i] w.r.t. control
        def fi_u(u):
            return dynamics_fn(state, u)[i]
        
        f_uu_i = hessian(fi_u)(control)
        f_uu = f_uu.at[i].set(f_uu_i)
        
        # f_ux[i]: Mixed Hessian of f[i] w.r.t. control and state
        # This is computed as: ∂²f[i]/∂u∂x = ∂(∂f[i]/∂u)/∂x
        # First compute ∂f[i]/∂u, then take its Jacobian w.r.t. x
        def fi_u_grad(s):
            # Compute gradient w.r.t. control
            def fi_val(u):
                return dynamics_fn(s, u)[i]
            return grad(fi_val)(control)
        
        f_ux_i = jacrev(fi_u_grad)(state)
        f_ux = f_ux.at[i].set(f_ux_i)
    
    return f_xx, f_uu, f_ux


def compute_cost_derivatives(running_cost_fn, terminal_cost_fn, state, control, is_terminal=False):
    """
    Compute cost derivatives: L_x, L_u, L_xx, L_uu
    
    Args:
        running_cost_fn: RunningCost instance (callable)
        terminal_cost_fn: TerminalCost instance (callable)
        state: [10] current state
        control: [6] current control (ignored if terminal)
        is_terminal: whether this is terminal cost
    
    Returns:
        L_x: [10] gradient w.r.t. state
        L_u: [6] gradient w.r.t. control (zero if terminal)
        L_xx: [10, 10] Hessian w.r.t. state
        L_uu: [6, 6] Hessian w.r.t. control (zero if terminal)
    """
    if is_terminal:
        # Terminal cost
        def cost_fn_terminal(s):
            return terminal_cost_fn(s)
        
        # Gradient and Hessian
        L_x = grad(cost_fn_terminal)(state)
        L_xx = hessian(cost_fn_terminal)(state)
        
        L_u = jnp.zeros(6)
        L_uu = jnp.zeros((6, 6))
        
    else:
        # Running cost
        def cost_fn_running(s, u):
            return running_cost_fn(s, u)
        
        # Gradient
        L_x = grad(lambda s: cost_fn_running(s, control))(state)
        L_u = grad(lambda u: cost_fn_running(state, u))(control)
        
        # Hessian
        L_xx = hessian(lambda s: cost_fn_running(s, control))(state)
        L_uu = hessian(lambda u: cost_fn_running(state, u))(control)
    
    return L_x, L_u, L_xx, L_uu


class DDP:
    """
    Differential Dynamic Programming / Iterative LQR Solver
    """
    
    def __init__(self, dynamics_model, running_cost, terminal_cost, 
                 max_iter=50, tol=1e-4, reg_init=1.0, reg_min=1e-6, reg_max=1e6,
                 reg_factor=10.0, line_search_alpha=0.5, line_search_beta=0.8, 
                 use_full_ddp=True):
        """
        Args:
            dynamics_model: SpaceRobotDynamics instance
            running_cost: RunningCost instance
            terminal_cost: TerminalCost instance
            max_iter: maximum iterations
            tol: convergence tolerance
            reg_init: initial regularization
            reg_min: minimum regularization
            reg_max: maximum regularization
            reg_factor: regularization update factor
            line_search_alpha: line search acceptance parameter
            line_search_beta: line search step size reduction
            use_full_ddp: if True, include dynamics curvature terms (slower but more accurate)
                         if False, use iLQR approximation (faster)
        """
        self.dynamics = dynamics_model
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.max_iter = max_iter
        self.tol = tol
        self.reg_init = reg_init
        self.reg_min = reg_min
        self.reg_max = reg_max
        self.reg_factor = reg_factor
        self.line_search_alpha = line_search_alpha
        self.line_search_beta = line_search_beta
        self.use_full_ddp = use_full_ddp
    
    def backward_pass(self, states, controls, dt, reg, use_full_ddp=True):
        """
        Backward pass: compute control gains k (feedforward) and K (feedback).
        
        Args:
            states: [T+1, 10] state trajectory
            controls: [T, 6] control sequence
            dt: time step
            reg: regularization parameter
            use_full_ddp: if True, include dynamics curvature terms (V_x @ f_xx, etc.)
                         if False, use iLQR approximation (faster but less accurate)
        
        Returns:
            k: [T, 6] feedforward gains
            K: [T, 6, 10] feedback gains
        """
        T = controls.shape[0]
        n_x = 10
        n_u = 6
        
        # Initialize value function derivatives
        V_x, V_xx = self._compute_terminal_value_derivatives(states[-1])
        
        # Storage for gains
        k = jnp.zeros((T, n_u))
        K = jnp.zeros((T, n_u, n_x))
        
        # Backward recursion
        for t in range(T - 1, -1, -1):
            # Current state and control
            x = states[t]
            u = controls[t]
            
            # Dynamics Jacobians
            f_x, f_u = compute_dynamics_jacobians(self.dynamics, x, u, dt)
            
            # Cost derivatives
            L_x, L_u, L_xx, L_uu = compute_cost_derivatives(
                self.running_cost, self.terminal_cost, x, u, is_terminal=False
            )
            
            # Q-function derivatives (quadratic approximation)
            Q_x = L_x + f_x.T @ V_x
            Q_u = L_u + f_u.T @ V_x
            Q_xx = L_xx + f_x.T @ V_xx @ f_x
            Q_uu = L_uu + f_u.T @ V_xx @ f_u
            Q_ux = f_u.T @ V_xx @ f_x
            
            # Add dynamics curvature terms for full DDP (if enabled)
            if use_full_ddp:
                f_xx, f_uu, f_ux = compute_dynamics_hessians(self.dynamics, x, u, dt)
                
                # Add curvature terms: Σᵢ V_x[i] * f_xx[i], etc.
                curvature_xx = jnp.zeros((n_x, n_x))
                curvature_uu = jnp.zeros((n_u, n_u))
                curvature_ux = jnp.zeros((n_u, n_x))
                
                for i in range(n_x):
                    curvature_xx += V_x[i] * f_xx[i]
                    curvature_uu += V_x[i] * f_uu[i]
                    curvature_ux += V_x[i] * f_ux[i]
                
                Q_xx = Q_xx + curvature_xx
                Q_uu = Q_uu + curvature_uu
                Q_ux = Q_ux + curvature_ux
            
            # Regularization for numerical stability
            Q_uu_reg = Q_uu + reg * jnp.eye(n_u)
            
            # Solve for gains
            try:
                Q_uu_inv = jnp.linalg.solve(Q_uu_reg, jnp.eye(n_u))
            except:
                # Fallback to pseudo-inverse if singular
                Q_uu_inv = jnp.linalg.pinv(Q_uu_reg)
            
            k = k.at[t].set(-Q_uu_inv @ Q_u)
            K = K.at[t].set(-Q_uu_inv @ Q_ux)
            
            # Update value function derivatives
            V_x = Q_x + K[t].T @ Q_uu @ k[t] + K[t].T @ Q_u + Q_ux.T @ k[t]
            V_xx = Q_xx + K[t].T @ Q_uu @ K[t] + K[t].T @ Q_ux + Q_ux.T @ K[t]
        
        return k, K
    
    def forward_pass(self, initial_state, nominal_states, controls, k, K, dt, alpha=1.0):
        """
        Forward pass: rollout new trajectory with updated controls.
        
        Args:
            initial_state: [10] initial state
            nominal_states: [T+1, 10] nominal state trajectory
            controls: [T, 6] nominal control sequence
            k: [T, 6] feedforward gains
            K: [T, 6, 10] feedback gains
            dt: time step
            alpha: line search step size
        
        Returns:
            new_states: [T+1, 10] new state trajectory
            new_controls: [T, 6] new control sequence
            total_cost: scalar total cost
        """
        T = controls.shape[0]
        new_states = jnp.zeros((T + 1, 10))
        new_controls = jnp.zeros((T, 6))
        new_states = new_states.at[0].set(initial_state)
        
        total_cost = 0.0
        
        def scan_step(carry, t):
            state_t, cost_accum = carry
            # Compute control update
            dx = state_t - nominal_states[t]  # Deviation from nominal
            du = alpha * k[t] + K[t] @ dx
            
            # Apply control (before safety measures)
            u_new = controls[t] + du

            # --------------------------------------------------------------
            # Numerical safety: sanitize and clamp controls to avoid blow-up
            # --------------------------------------------------------------
            # Replace NaNs / Infs with finite values
            u_new = jnp.nan_to_num(u_new, nan=0.0, posinf=1e3, neginf=-1e3)
            # Hard bound on joint velocities (rad/s)
            u_max = 1.0
            u_new = jnp.clip(u_new, -u_max, u_max)
            
            # Step dynamics
            state_next = self.dynamics.step(state_t, u_new, dt)
            # Sanitize state to prevent NaN propagation
            state_next = jnp.nan_to_num(state_next, nan=0.0, posinf=1e3, neginf=-1e3)
            
            # Accumulate cost
            cost_new = self.running_cost(state_t, u_new)
            cost_accum = cost_accum + cost_new
            
            return (state_next, cost_accum), (state_next, u_new)
        
        # Use scan for efficient computation
        (final_state, total_cost), (states_array, controls_array) = jax.lax.scan(
            scan_step, (initial_state, 0.0), jnp.arange(T)
        )
        
        # Stack initial state
        new_states = jnp.vstack([initial_state, states_array])
        new_controls = controls_array
        
        # Terminal cost
        total_cost = total_cost + self.terminal_cost(new_states[-1])
        
        return new_states, new_controls, total_cost
    
    def _compute_terminal_value_derivatives(self, terminal_state):
        """Compute terminal value function derivatives."""
        L_x, _, L_xx, _ = compute_cost_derivatives(
            self.running_cost, self.terminal_cost, terminal_state, 
            jnp.zeros(6), is_terminal=True
        )
        return L_x, L_xx
    
    def solve(self, initial_state, initial_controls, dt):
        """
        Solve DDP optimization problem.

        Args:
            initial_state: [10] initial state
            initial_controls: [T, 6] initial control sequence
            dt: time step

        Returns:
            states: [T+1, 10] optimal state trajectory
            controls: [T, 6] optimal control sequence
            cost_history: list of costs per iteration
        """
        # Initial rollout
        states = self.dynamics.rollout(initial_state, initial_controls, dt)
        controls = initial_controls

        # Compute initial cost
        initial_cost = 0.0
        for t in range(controls.shape[0]):
            initial_cost = initial_cost + self.running_cost(states[t], controls[t])
        initial_cost = initial_cost + self.terminal_cost(states[-1])

        cost_history = [float(initial_cost)]

        reg = self.reg_init

        print(f"Iteration 0: Initial cost = {initial_cost:.6f}, reg = {reg:.6e}")

        # Main iteration loop (kept in Python for clarity and to avoid JAX
        # host/device value conversion issues from Python floats and lists)
        for iteration in range(self.max_iter):
            # Backward pass
            k, K = self.backward_pass(states, controls, dt, reg, self.use_full_ddp)

            # Forward pass with line search
            alpha = 1.0
            best_cost = float('inf')
            best_states = None
            best_controls = None
            best_alpha = 0.0

            for ls_iter in range(10):  # Max line search iterations
                new_states, new_controls, new_cost = self.forward_pass(
                    initial_state, states, controls, k, K, dt, alpha
                )

                new_cost_val = float(new_cost)
                if new_cost_val < best_cost:
                    best_cost = new_cost_val
                    best_states = new_states
                    best_controls = new_controls
                    best_alpha = alpha

                    # Check improvement
                    improvement = cost_history[-1] - best_cost
                    k_norm_sq = float(jnp.sum(k ** 2))
                    if improvement >= self.line_search_alpha * alpha * k_norm_sq:
                        break

                alpha *= self.line_search_beta

            # Update trajectory
            if best_states is not None:
                prev_cost = cost_history[-1]
                cost_reduction = prev_cost - best_cost

                states = best_states
                controls = best_controls
                cost_history.append(best_cost)

                # Print iteration info
                print(f"Iteration {iteration + 1}: Cost = {best_cost:.6f}, "
                      f"Reduction = {cost_reduction:.6e}, "
                      f"Alpha = {best_alpha:.4f}, "
                      f"Reg = {reg:.6e}")

                # Check convergence
                if len(cost_history) > 1:
                    if cost_reduction < self.tol:
                        print(f"Converged! Cost reduction ({cost_reduction:.6e}) < tolerance ({self.tol:.6e})")
                        break

                # Update regularization
                if best_cost < cost_history[-2] if len(cost_history) > 1 else True:
                    reg = max(reg / self.reg_factor, self.reg_min)
                else:
                    reg = min(reg * self.reg_factor, self.reg_max)
            else:
                # No improvement, increase regularization
                reg = min(reg * self.reg_factor, self.reg_max)
                print(f"Iteration {iteration + 1}: No improvement, reg = {reg:.6e}")

        return states, controls, cost_history

