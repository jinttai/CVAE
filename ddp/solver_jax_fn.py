"""
Functional JAX iLQR / DDP-style solver.

This module provides a pure-function style solver that is easier to JIT.
Current implementation focuses on iLQR (no dynamics curvature terms).
"""

import jax
import jax.numpy as jnp
from jax import grad, jacrev, hessian


def _compute_dynamics_jacobians(dynamics_model, state, control, dt):
    """
    Compute dynamics Jacobians: f_x, f_u
    using the provided JAX dynamics_model.
    """

    def dynamics_fn(s, u):
        return dynamics_model.step(s, u, dt)

    f_x = jacrev(lambda s: dynamics_fn(s, control))(state)
    f_u = jacrev(lambda u: dynamics_fn(state, u))(control)
    return f_x, f_u


def _compute_cost_derivatives(running_cost_fn, terminal_cost_fn, state, control, is_terminal=False):
    """
    Pure JAX cost derivatives: L_x, L_u, L_xx, L_uu
    """
    if is_terminal:
        def cost_fn_terminal(s):
            return terminal_cost_fn(s)

        L_x = grad(cost_fn_terminal)(state)
        L_xx = hessian(cost_fn_terminal)(state)

        L_u = jnp.zeros_like(control)
        L_uu = jnp.zeros((control.shape[0], control.shape[0]))
    else:
        def cost_fn_running(s, u):
            return running_cost_fn(s, u)

        L_x = grad(lambda s: cost_fn_running(s, control))(state)
        L_u = grad(lambda u: cost_fn_running(state, u))(control)

        L_xx = hessian(lambda s: cost_fn_running(s, control))(state)
        L_uu = hessian(lambda u: cost_fn_running(state, u))(control)

    return L_x, L_u, L_xx, L_uu


def _rollout(dynamics_model, initial_state, controls, dt):
    """
    Rollout using the dynamics_model (which already has jitted step/rollout).
    """
    return dynamics_model.rollout(initial_state, controls, dt)


def _total_cost(running_cost_fn, terminal_cost_fn, states, controls):
    """
    Compute total cost along a trajectory (pure JAX).
    """
    T = controls.shape[0]

    def body_fun(t, accum):
        return accum + running_cost_fn(states[t], controls[t])

    total = jax.lax.fori_loop(0, T, body_fun, 0.0)
    total = total + terminal_cost_fn(states[-1])
    return total


def ilqr_solve(
    initial_state,
    initial_controls,
    dt,
    dynamics_model,
    running_cost,
    terminal_cost,
    max_iter=50,
    reg_init=1.0,
    reg_min=1e-6,
    reg_max=1e6,
    reg_factor=10.0,
):
    """
    Functional iLQR solver (no dynamics curvature terms, i.e., use_full_ddp=False).

    This function is written in a JAX-friendly style (no Python floats/lists
    inside the core computations) so that it can be JIT-compiled.

    Args:
        initial_state: [n_x] initial state
        initial_controls: [T, n_u] initial control sequence
        dt: time step (scalar)
        dynamics_model: SpaceRobotDynamics instance (used as static arg when jitting)
        running_cost: callable(state, control) -> scalar
        terminal_cost: callable(state) -> scalar
        max_iter: maximum iLQR iterations
        reg_init, reg_min, reg_max, reg_factor: Levenberg-Marquardt style regularization

    Returns:
        states: [T+1, n_x]
        controls: [T, n_u]
        cost_history: [max_iter+1]
    """
    n_x = initial_state.shape[0]
    n_u = initial_controls.shape[1]
    T = initial_controls.shape[0]

    # Initial rollout
    states = _rollout(dynamics_model, initial_state, initial_controls, dt)
    controls = initial_controls

    # Initial cost
    initial_cost = _total_cost(running_cost, terminal_cost, states, controls)

    # Cost history as JAX array
    cost_history = jnp.full((max_iter + 1,), jnp.inf)
    cost_history = cost_history.at[0].set(initial_cost)

    reg = reg_init

    def iteration_body(iter_idx, carry):
        states, controls, cost_history, reg = carry

        # Check current trajectory before doing backward pass
        nan_state_prev = jnp.any(jnp.isnan(states))
        nan_ctrl_prev = jnp.any(jnp.isnan(controls))
        max_state_prev = jnp.max(jnp.abs(states))
        max_ctrl_prev = jnp.max(jnp.abs(controls))
        jax.debug.print(
            "iter {i} (before backward): nan_state={ns}, nan_ctrl={nc}, "
            "max_state={ms}, max_ctrl={mc}",
            i=iter_idx,
            ns=nan_state_prev,
            nc=nan_ctrl_prev,
            ms=max_state_prev,
            mc=max_ctrl_prev,
        )

        # Terminal value derivatives
        L_x_T, _, L_xx_T, _ = _compute_cost_derivatives(
            running_cost, terminal_cost, states[-1], jnp.zeros(n_u), is_terminal=True
        )
        V_x = L_x_T
        V_xx = L_xx_T

        # Backward pass (iLQR: no dynamics curvature terms)
        k = jnp.zeros((T, n_u))
        K = jnp.zeros((T, n_u, n_x))

        def backward_step(t, bk_carry):
            V_x, V_xx, k, K = bk_carry
            idx = T - 1 - t  # reverse time

            x = states[idx]
            u = controls[idx]

            f_x, f_u = _compute_dynamics_jacobians(dynamics_model, x, u, dt)
            L_x, L_u, L_xx, L_uu = _compute_cost_derivatives(
                running_cost, terminal_cost, x, u, is_terminal=False
            )

            Q_x = L_x + f_x.T @ V_x
            Q_u = L_u + f_u.T @ V_x
            Q_xx = L_xx + f_x.T @ V_xx @ f_x
            Q_uu = L_uu + f_u.T @ V_xx @ f_u
            Q_ux = f_u.T @ V_xx @ f_x

            Q_uu_reg = Q_uu + reg * jnp.eye(n_u)
            Q_uu_inv = jnp.linalg.pinv(Q_uu_reg)

            k_t = -Q_uu_inv @ Q_u
            K_t = -Q_uu_inv @ Q_ux

            k = k.at[idx].set(k_t)
            K = K.at[idx].set(K_t)

            V_x_new = Q_x + K_t.T @ Q_uu @ k_t + K_t.T @ Q_u + Q_ux.T @ k_t
            V_xx_new = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_ux.T @ K_t

            return (V_x_new, V_xx_new, k, K)

        V_x, V_xx, k, K = jax.lax.fori_loop(0, T, backward_step, (V_x, V_xx, k, K))

        # Forward pass with fixed alpha = 1.0 (no line search for simplicity)
        def forward_step(carry_fwd, t):
            state_t, states_new, controls_new = carry_fwd

            dx = state_t - states[t]
            du = k[t] + K[t] @ dx

            # Updated control before safety processing
            u_new = controls[t] + du

            # Clamp controls to avoid unbounded velocities (still JIT-friendly)
            u_new = jnp.nan_to_num(u_new, nan=0.0, posinf=1e3, neginf=-1e3)
            u_max = 1.0  # rad/s bound, can be tuned
            u_new = jnp.clip(u_new, -u_max, u_max)

            state_next = dynamics_model.step(state_t, u_new, dt)

            states_new = states_new.at[t + 1].set(state_next)
            controls_new = controls_new.at[t].set(u_new)

            return (state_next, states_new, controls_new), None

        states_new = jnp.zeros_like(states)
        controls_new = jnp.zeros_like(controls)
        states_new = states_new.at[0].set(initial_state)

        (final_state, states_new, controls_new), _ = jax.lax.scan(
            forward_step, (initial_state, states_new, controls_new), jnp.arange(T)
        )

        new_cost = _total_cost(running_cost, terminal_cost, states_new, controls_new)

        # Debug: check for NaNs and large values
        nan_state = jnp.any(jnp.isnan(states_new))
        nan_ctrl = jnp.any(jnp.isnan(controls_new))
        nan_cost = jnp.isnan(new_cost)
        max_state = jnp.max(jnp.abs(states_new))
        max_ctrl = jnp.max(jnp.abs(controls_new))

        jax.debug.print(
            "iter {i}: nan_state={ns}, nan_ctrl={nc}, nan_cost={ncost}, "
            "max_state={ms}, max_ctrl={mc}, cost={c}",
            i=iter_idx,
            ns=nan_state,
            nc=nan_ctrl,
            ncost=nan_cost,
            ms=max_state,
            mc=max_ctrl,
            c=new_cost,
        )

        cost_history = cost_history.at[iter_idx + 1].set(new_cost)

        # Simple regularization update based on cost improvement
        prev_cost = cost_history[iter_idx]
        cost_reduction = prev_cost - new_cost
        reg_new = jnp.where(
            cost_reduction > 0.0,
            jnp.maximum(reg / reg_factor, reg_min),
            jnp.minimum(reg * reg_factor, reg_max),
        )

        return (states_new, controls_new, cost_history, reg_new)

    # Run fixed number of iterations with fori_loop
    states, controls, cost_history, reg = jax.lax.fori_loop(
        0, max_iter, iteration_body, (states, controls, cost_history, reg)
    )

    return states, controls, cost_history


# JIT-compiled version, treating dynamics_model and cost objects as static.
ilqr_solve_jit = jax.jit(
    ilqr_solve,
    static_argnames=(
        "dynamics_model",
        "running_cost",
        "terminal_cost",
        "max_iter",
        "reg_init",
        "reg_min",
        "reg_max",
        "reg_factor",
    ),
)


