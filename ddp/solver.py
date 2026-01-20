"""
DDP/iLQR Solver
PyTorch implementation of Differential Dynamic Programming / Iterative LQR.
"""

import torch
from torch.func import jacrev, hessian
import ddp.dynamics_torch as dynamics
import ddp.cost as cost


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
    state = state.clone().requires_grad_(True)
    control = control.clone().requires_grad_(True)
    
    def dynamics_fn(s, u):
        return dynamics_model.step(s, u, dt)
    
    # Compute Jacobians using torch.func
    f_x = jacrev(lambda s: dynamics_fn(s, control), has_aux=False)(state)
    f_u = jacrev(lambda u: dynamics_fn(state, u), has_aux=False)(control)
    
    return f_x, f_u


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
    state = state.clone().requires_grad_(True)
    
    if is_terminal:
        # Terminal cost
        def cost_fn_terminal(s):
            return terminal_cost_fn(s)
        
        # Gradient
        L_x = torch.autograd.grad(
            cost_fn_terminal(state),
            state,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Hessian
        L_xx = torch.zeros(10, 10, device=state.device)
        for i in range(10):
            grad_i = torch.autograd.grad(
                L_x[i],
                state,
                retain_graph=(i < 9),
                create_graph=False
            )[0]
            L_xx[i] = grad_i
        
        L_u = torch.zeros(6, device=state.device)
        L_uu = torch.zeros(6, 6, device=state.device)
        
    else:
        # Running cost
        control = control.clone().requires_grad_(True)
        
        def cost_fn_running(s, u):
            return running_cost_fn(s, u)
        
        # Gradient
        L_x = torch.autograd.grad(
            cost_fn_running(state, control),
            state,
            create_graph=True,
            retain_graph=True
        )[0]
        
        L_u = torch.autograd.grad(
            cost_fn_running(state, control),
            control,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Hessian (approximate using second-order derivatives)
        L_xx = torch.zeros(10, 10, device=state.device)
        for i in range(10):
            grad_i = torch.autograd.grad(
                L_x[i],
                state,
                retain_graph=(i < 9),
                create_graph=False
            )[0]
            L_xx[i] = grad_i
        
        L_uu = torch.zeros(6, 6, device=state.device)
        for i in range(6):
            grad_i = torch.autograd.grad(
                L_u[i],
                control,
                retain_graph=(i < 5),
                create_graph=False
            )[0]
            L_uu[i] = grad_i
    
    return L_x, L_u, L_xx, L_uu


class DDP:
    """
    Differential Dynamic Programming / Iterative LQR Solver
    """
    
    def __init__(self, dynamics_model, running_cost, terminal_cost, 
                 max_iter=50, tol=1e-4, reg_init=1.0, reg_min=1e-6, reg_max=1e6,
                 reg_factor=10.0, line_search_alpha=0.5, line_search_beta=0.8, device='cpu'):
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
        
        self.device = device if device else dynamics_model.device
    
    def backward_pass(self, states, controls, dt, reg):
        """
        Backward pass: compute control gains k (feedforward) and K (feedback).
        
        Args:
            states: [T+1, 10] state trajectory
            controls: [T, 6] control sequence
            dt: time step
            reg: regularization parameter
        
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
        k = torch.zeros(T, n_u, device=self.device)
        K = torch.zeros(T, n_u, n_x, device=self.device)
        
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
            
            # Regularization for numerical stability
            Q_uu_reg = Q_uu + reg * torch.eye(n_u, device=self.device)
            
            # Solve for gains
            try:
                Q_uu_inv = torch.linalg.solve(Q_uu_reg, torch.eye(n_u, device=self.device))
            except:
                # Fallback to pseudo-inverse if singular
                Q_uu_inv = torch.linalg.pinv(Q_uu_reg)
            
            k[t] = -Q_uu_inv @ Q_u
            K[t] = -Q_uu_inv @ Q_ux
            
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
        new_states = torch.zeros(T + 1, 10, device=self.device)
        new_controls = torch.zeros(T, 6, device=self.device)
        new_states[0] = initial_state
        
        total_cost = 0.0
        
        for t in range(T):
            # Compute control update
            dx = new_states[t] - nominal_states[t]  # Deviation from nominal
            du = alpha * k[t] + K[t] @ dx
            
            # Apply control
            new_controls[t] = controls[t] + du
            
            # Step dynamics
            new_states[t + 1] = self.dynamics.step(new_states[t], new_controls[t], dt)
            
            # Accumulate cost
            total_cost = total_cost + self.running_cost(new_states[t], new_controls[t])
        
        # Terminal cost
        total_cost = total_cost + self.terminal_cost(new_states[-1])
        
        return new_states, new_controls, total_cost
    
    def _compute_terminal_value_derivatives(self, terminal_state):
        """Compute terminal value function derivatives."""
        L_x, _, L_xx, _ = compute_cost_derivatives(
            self.running_cost, self.terminal_cost, terminal_state, 
            torch.zeros(6, device=self.device), is_terminal=True
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
        controls = initial_controls.clone()
        
        # Compute initial cost
        initial_cost = 0.0
        for t in range(controls.shape[0]):
            initial_cost = initial_cost + self.running_cost(states[t], controls[t])
        initial_cost = initial_cost + self.terminal_cost(states[-1])
        
        cost_history = [initial_cost.item()]
        
        reg = self.reg_init
        
        # Main iteration loop
        for iteration in range(self.max_iter):
            # Backward pass
            k, K = self.backward_pass(states, controls, dt, reg)
            
            # Forward pass with line search
            alpha = 1.0
            best_cost = float('inf')
            best_states = None
            best_controls = None
            
            for _ in range(10):  # Max line search iterations
                new_states, new_controls, new_cost = self.forward_pass(
                    initial_state, states, controls, k, K, dt, alpha
                )
                
                if new_cost.item() < best_cost:
                    best_cost = new_cost.item()
                    best_states = new_states
                    best_controls = new_controls
                    
                    # Check improvement
                    improvement = cost_history[-1] - best_cost
                    if improvement >= self.line_search_alpha * alpha * torch.sum(k ** 2).item():
                        break
                
                alpha *= self.line_search_beta
            
            # Update trajectory
            if best_states is not None:
                states = best_states
                controls = best_controls
                cost_history.append(best_cost)
                
                # Check convergence
                if len(cost_history) > 1:
                    cost_reduction = cost_history[-2] - cost_history[-1]
                    if cost_reduction < self.tol:
                        break
                
                # Update regularization
                if best_cost < cost_history[-2] if len(cost_history) > 1 else True:
                    reg = max(reg / self.reg_factor, self.reg_min)
                else:
                    reg = min(reg * self.reg_factor, self.reg_max)
            else:
                # No improvement, increase regularization
                reg = min(reg * self.reg_factor, self.reg_max)
        
        return states, controls, cost_history

