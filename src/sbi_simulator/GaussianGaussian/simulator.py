"""Simulator for the Gaussian-Gaussian model.
reference: 'Compositional Score Modeling for Simulation-Based Inference'
"""

from typing import override

import pyro.distributions as pdist
import torch

from ..Simulator import Simulator


class GaussianGaussian(Simulator):
    def __init__(
        self,
        dim_parameters: int = 10,
        seed: int = 42,
        dim_data: int | None = None,
    ):
        if dim_data is None:
            dim_data = dim_parameters
        super().__init__(seed=seed, dim_parameters=dim_parameters, dim_data=dim_data)

    def _diag_sigma(self, dim: int) -> torch.Tensor:
        # Variances scaling linearly from 0.6 to 1.4 across `dim` entries
        if dim == 1:
            return torch.tensor([0.6])
        return 0.6 + torch.arange(dim, dtype=torch.float32) * (1.4 - 0.6) / (dim - 1)

    @override
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        params = torch.as_tensor(parameters, dtype=torch.float32)
        if params.numel() == 0:
            raise ValueError("Expected at least one parameter.")

        is_1d = params.ndim == 1
        if is_1d:
            params = params.unsqueeze(0)

        dim = params.shape[-1]
        diag_sigma = self._diag_sigma(dim).to(params.device)
        std = torch.sqrt(diag_sigma)

        noise = torch.randn_like(params) * std
        out = params + noise
        return out.squeeze(0) if is_1d else out

    @override
    def get_prior(self) -> pdist.Distribution:
        return pdist.Normal(
            torch.zeros(self.dim_parameters), torch.ones(self.dim_parameters)
        ).to_event(1)

    def posterior_samples(
        self,
        x_obs: torch.Tensor | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the analytic posterior for the Gaussian/Gaussian simulator
        with prior N(0, I) and likelihood N(x | theta, Sigma), Sigma diagonal.
        Supports single observation (1D or 1xDim) or multiple observations (NxDim).

        Posterior: theta | x_1...x_N ~ N(mu_post, Sigma_post)
            Sigma_post = (I + N * Sigma^-1)^-1
            mu_post    = Sigma_post @ Sigma^-1 @ sum_{i=1}^N x_i
        """
        if (
            x_obs is None
            and "num_samples" in kwargs
            and isinstance(num_samples, torch.Tensor)
        ):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)
        if x_obs is None:
            x_obs = torch.zeros(self.dim_data, dtype=torch.float32)

        x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_t.ndim == 1:
            x_obs_t = x_obs_t.unsqueeze(0)

        n_obs, dim = x_obs_t.shape
        diag_sigma = self._diag_sigma(dim)
        sigma_inv = torch.diag(1.0 / diag_sigma)

        prior_precision = torch.eye(dim)
        posterior_precision = prior_precision + n_obs * sigma_inv
        posterior_cov = torch.linalg.inv(posterior_precision)
        posterior_mean = posterior_cov @ (sigma_inv @ x_obs_t.sum(dim=0))

        posterior = torch.distributions.MultivariateNormal(
            loc=posterior_mean, covariance_matrix=posterior_cov
        )
        return posterior.sample((num_samples,))
