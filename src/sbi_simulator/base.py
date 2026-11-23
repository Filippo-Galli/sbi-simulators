"""Base simulator abstraction for Simulation-Based Inference benchmarks."""

from abc import ABC, abstractmethod

import numpy as np
import pyro
import pyro.distributions as pdist
import torch


class Simulator(ABC):
    """Abstract base class for SBI benchmark simulators."""

    def __init__(
        self,
        seed: int = 42,
        dim_parameters: int = 2,
        dim_data: int | None = None,
        **kwargs,
    ):
        self.seed = seed
        self.dim_parameters = dim_parameters
        self.dim_data = dim_data if dim_data is not None else dim_parameters
        self.kwargs = kwargs
        self._np_rng = np.random.default_rng(seed)
        pyro.set_rng_seed(seed)

    def _prepare_parameters(
        self, parameters: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, bool]:
        """Convert parameters to float32 2D tensor and report whether input was 1D."""
        params = torch.as_tensor(parameters, dtype=torch.float32)
        if params.numel() == 0:
            raise ValueError("Expected at least one parameter.")
        is_1d = params.ndim == 1
        if is_1d:
            params = params.unsqueeze(0)
        return params, is_1d

    def sample_prior(self, num_samples: int = 1000) -> torch.Tensor:
        """Draw samples from the simulator prior."""
        return self.get_prior().sample((num_samples,))

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset internal PRNG state for reproducibility."""
        if seed is not None:
            self.seed = seed
        self._np_rng = np.random.default_rng(self.seed)
        pyro.set_rng_seed(self.seed)

    @abstractmethod
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor: ...

    @abstractmethod
    def get_prior(self) -> pdist.Distribution: ...

    @abstractmethod
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor: ...
