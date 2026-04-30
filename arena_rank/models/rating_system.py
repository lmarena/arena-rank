"""Base class for rating systems."""

from abc import ABC, abstractmethod
from functools import partial
from jaxtyping import PyTree

from arena_rank.utils.data_utils import BasePairDataset
from arena_rank.utils.math_utils import lbfgs_minimize


class RatingSystem(ABC):
    """Abstract base class for rating systems with differentiable loss functions optimized via L-BFGS in jax"""

    data: PyTree
    params: PyTree
    max_iter: int = 1000
    gtol: float = 1e-9
    ftol: float = 1e-9
    verbose: bool = False
    fitted: bool = False

    @abstractmethod
    def loss_function(self, params: PyTree, data: PyTree) -> float:
        """Each rating system must implement its own loss function."""

    @abstractmethod
    def compute_ratings_and_cis(self, dataset: BasePairDataset, alpha: float = 0.95) -> PyTree:
        """Compute ratings and confidence intervals given a dataset."""

    def fit(self, dataset: BasePairDataset):
        """Fit the rating system to the provided data."""
        initial_params = self.params
        opt_fn = partial(self.loss_function, data=dataset.as_dict())
        final_params, _ = lbfgs_minimize(
            function=opt_fn,
            initial_params=initial_params,
            max_iter=self.max_iter,
            gtol=self.gtol,
            ftol=self.ftol,
            verbose=self.verbose,
        )
        self.params = final_params
        self.fitted = True
        return self
