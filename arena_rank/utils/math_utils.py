"""
Utility functions for mathematical operations.
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import PyTree
import optax


@jit
def assemble_parwise_matrix(mat, vals, idx_a, idx_b):
    """
    Accumulates row-level values into a parameter-level pairwise matrix for Hessian and gradient covariance calculations.
    Args:
        mat: The matrix to accumulate into, shape (n_competitors, n_competitors)
        vals: The values to accumulate, shape (n_pairs,)
        idx_a: Indices of competitor A in each pair, shape (n_pairs,)
        idx_b: Indices of competitor B in each pair, shape (n_pairs,)
    Returns:
        The updated matrix with accumulated values.
    """
    mat = mat.at[idx_a, idx_a].add(vals)
    mat = mat.at[idx_b, idx_b].add(vals)
    mat = mat.at[idx_a, idx_b].add(-vals)
    mat = mat.at[idx_b, idx_a].add(-vals)
    return mat


def lbfgs_minimize(
    function: Callable[[PyTree], jnp.ndarray],
    initial_params: PyTree,
    max_iter: int = 1000,
    gtol: float = 1e-6,
    ftol: float = 1e-9,
    verbose: bool = False,
) -> Tuple[PyTree, PyTree]:
    """Minimize a differentiable function using the L-BFGS algorithm from Optax."""
    solver = optax.lbfgs()

    @jit
    def _run_optimization(init_params):
        value_and_grad_fun = optax.value_and_grad_from_state(function)

        # carry state: (params, solver_state, prev_loss, grad_inf_norm, rel_change)
        def step(carry: Tuple) -> Tuple:
            params, state, prev_loss, _, _ = carry
            value, grad = value_and_grad_fun(params, state=state)
            updates, state = solver.update(grad, state, params, value=value, grad=grad, value_fn=function)
            params = optax.apply_updates(params, updates)

            # extract metrics from state
            current_loss = optax.tree_utils.tree_get(state, "value", jnp.inf)
            current_grad = optax.tree_utils.tree_get(state, "grad", jnp.inf)

            # calculate infinity norm of grad to check against gtol
            leaf_maxes = jax.tree.map(lambda x: jnp.max(jnp.abs(x)), current_grad)
            grad_inf_norm = jax.tree_util.tree_reduce(jnp.maximum, leaf_maxes)

            # calculate relative loss change to check against ftol
            loss_change = jnp.abs(current_loss - prev_loss)
            max_loss = jnp.maximum(jnp.abs(current_loss), jnp.abs(prev_loss))
            denom = jnp.maximum(max_loss, 1.0)

            # handle first iteration where prev_loss is inf
            raw_rel_change = loss_change / denom
            rel_change = jnp.where(jnp.isinf(prev_loss), jnp.inf, raw_rel_change)

            return params, state, value, grad_inf_norm, rel_change

        def continuing_criterion(carry):
            _, state, _, grad_inf_norm, rel_change = carry

            # extract metrics from state
            iter_num = optax.tree_utils.tree_get(state, "count", 0)

            is_first_iter = iter_num == 0
            is_within_bounds = iter_num < max_iter
            is_not_converged = (grad_inf_norm >= gtol) & (rel_change >= ftol)
            return is_first_iter | (is_within_bounds & is_not_converged)

        # init with inf for metrics so loop starts correctly
        init_carry = (init_params, solver.init(init_params), jnp.inf, jnp.inf, jnp.inf)
        final_carry = jax.lax.while_loop(continuing_criterion, step, init_carry)
        return final_carry

    final_params, final_state, _, final_grad_norm, final_rel_change = _run_optimization(initial_params)

    # needed for accurate speed benchmarks
    # final_params = jax.tree.map(lambda x: x.block_until_ready(), final_params)

    final_iter = optax.tree_utils.tree_get(final_state, "count", 0)
    final_loss = optax.tree_utils.tree_get(final_state, "value", jnp.inf)

    if verbose:
        print(f"L-BFGS finished in {final_iter} iterations.")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Rel F Diff: {final_rel_change.item():.2e} (tol={ftol})")
        print(f"  Grad Norm:  {final_grad_norm.item():.2e} (tol={gtol})")

    return final_params, final_state
