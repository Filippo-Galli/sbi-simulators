from abc import ABC, abstractmethod

import pyro
import pyro.distributions as pdist
import torch


class Simulator(ABC):
    def __init__(
        self,
        seed: int = 42,
        dim_parameters: int = 2,
        dim_data: int | None = None,
    ):
        self.seed = seed
        self.dim_parameters = dim_parameters
        self.dim_data = dim_data if dim_data is not None else dim_parameters
        pyro.set_rng_seed(seed)

    @abstractmethod
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def get_prior(self) -> pdist.Distribution: ...
