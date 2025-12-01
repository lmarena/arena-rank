"""Base class for rating systems."""

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import PyTree
import optax


class RatingSystem(ABC):
    """Abstract base class for rating systems."""

    @abstractmethod
    def fit(self, data, labels):
        """Fit the rating system to the provided data."""
        pass

    def lbfgs_minimize(
        self,
        loss_function: Callable[[PyTree, PyTree], jnp.ndarray],
        initial_params: PyTree,
        max_iter: int = 1000,
        gtol: float = 1e-6,
        ftol: float = 1e-9,
    ) -> Tuple[PyTree, PyTree]:
        # Initialize the solver
        solver = optax.lbfgs()

        @jit
        def _run_optimization(init_params):
            value_and_grad_fun = optax.value_and_grad_from_state(loss_function)

            # Carry state: (params, solver_state, prev_loss)
            def step(carry: Tuple) -> Tuple:
                params, state, _ = carry
                # The current value becomes prev_loss for the next iteration logic
                value, grad = value_and_grad_fun(params, state=state)
                updates, state = solver.update(grad, state, params, value=value, grad=grad, value_fn=loss_function)
                params = optax.apply_updates(params, updates)
                return params, state, value

            def continuing_criterion(carry):
                _, state, prev_loss = carry

                # Extract metrics from state
                iter_num = optax.tree_utils.tree_get(state, "count", 0)
                current_loss = optax.tree_utils.tree_get(state, "value", jnp.inf)
                grad = optax.tree_utils.tree_get(state, "grad", jnp.inf)

                # 1. Gradient Infinity Norm Check
                # Calculate max(abs(x)) for every leaf, then reduce to global max
                leaf_maxes = jax.tree.map(lambda x: jnp.max(jnp.abs(x)), grad)
                grad_inf_norm = jax.tree_util.tree_reduce(jnp.maximum, leaf_maxes)

                # 2. Relative Function Difference Check
                loss_change = jnp.abs(current_loss - prev_loss)
                max_loss = jnp.maximum(jnp.abs(current_loss), jnp.abs(prev_loss))
                denom = jnp.maximum(max_loss, 1.0)
                rel_change = loss_change / denom

                # 3. Control Logic
                is_first_iter = iter_num == 0
                is_within_bounds = iter_num < max_iter
                is_not_converged = (grad_inf_norm >= gtol) & (rel_change >= ftol)

                # Continue if it's the first run OR (we haven't hit max_iter AND we haven't converged)
                return is_first_iter | (is_within_bounds & is_not_converged)

            # Initialize with inf for prev_loss so ftol doesn't trigger on step 0
            init_carry = (init_params, solver.init(init_params), jnp.inf)

            # Run the loop
            final_carry = jax.lax.while_loop(continuing_criterion, step, init_carry)
            return final_carry

        # Execute JIT-compiled optimization
        final_params, final_state, final_prev_loss = _run_optimization(initial_params)

        # --- Synchronization ---
        # Force host to wait for device computation to finish.
        # Crucial for accurate timing in external scripts.
        final_params = jax.tree.map(lambda x: x.block_until_ready(), final_params)

        # --- Logging ---
        # Extract metrics (scalars) to CPU for printing
        final_iter = optax.tree_utils.tree_get(final_state, "count", 0)
        final_loss = optax.tree_utils.tree_get(final_state, "value", jnp.inf)
        final_grad = optax.tree_utils.tree_get(final_state, "grad", initial_params)

        # Calculate final metrics for display
        # 1. Grad Norm (Infinity)
        grad_leaves = jax.tree_util.tree_leaves(final_grad)
        # Handle case where grad might be empty or invalid structure
        if grad_leaves and hasattr(grad_leaves[0], "shape"):
            grad_norm = max([jnp.max(jnp.abs(x)).item() for x in grad_leaves])
        else:
            grad_norm = float("inf")

        # 2. Relative Function Diff
        loss_diff = jnp.abs(final_loss - final_prev_loss).item()
        denom = max(abs(final_loss), abs(final_prev_loss), 1.0)
        rel_change = loss_diff / denom

        print(f"L-BFGS finished in {final_iter} iterations.")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Rel F Diff: {rel_change:.2e} (tol={ftol})")
        print(f"  Grad Norm:  {grad_norm:.2e} (tol={gtol})")

        return final_params, final_state
