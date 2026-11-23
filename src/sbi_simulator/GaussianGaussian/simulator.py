"""Simulator for the Gaussian-Gaussian model.
reference: 'Compositional Score Modeling for Simulation-Based Inference'
"""

import math
from typing import override

import pyro.distributions as pdist
import torch

from ..Simulator import Simulator


class GaussianGaussian(Simulator):
    """Simulator for the Gaussian-Gaussian / Gaussian Linear model.

    Presets:
    --------
    - "default":
      Configuration from 'Compositional Score Modeling for Simulation-Based Inference':
      * Prior: N(0, I_d)
      * Likelihood: N(x | theta, diag_sigma), variances linearly spaced in [0.6, 1.4]
    - "sbibm" (Lueckmann et al., 2021, arXiv:2101.04653, Task 'gaussian_linear'):
      * Prior: N(0, 0.1 * I_d)
      * Likelihood: N(x | theta, 0.1 * I_d)
    """

    def __init__(
        self,
        dim_parameters: int = 10,
        seed: int = 42,
        dim_data: int | None = None,
        preset: str = "default",
        **kwargs,
    ):
        preset_key = preset.lower()
        if preset_key not in ("default", "sbibm"):
            raise ValueError(
                f"Unknown preset '{preset}'. Supported presets are 'default' and 'sbibm'."
            )
        self.preset = preset_key
        if dim_data is None:
            dim_data = dim_parameters
        super().__init__(
            seed=seed, dim_parameters=dim_parameters, dim_data=dim_data, **kwargs
        )

    def _diag_sigma(self, dim: int) -> torch.Tensor:
        if self.preset == "sbibm":
            return torch.full((dim,), 0.1, dtype=torch.float32)
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
        std = math.sqrt(0.1) if self.preset == "sbibm" else 1.0
        return pdist.Normal(
            torch.zeros(self.dim_parameters),
            std * torch.ones(self.dim_parameters),
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

        prior_var = 0.1 if self.preset == "sbibm" else 1.0
        prior_precision = (1.0 / prior_var) * torch.eye(dim)
        posterior_precision = prior_precision + n_obs * sigma_inv
        posterior_cov = torch.linalg.inv(posterior_precision)
        posterior_mean = posterior_cov @ (sigma_inv @ x_obs_t.sum(dim=0))

        posterior = torch.distributions.MultivariateNormal(
            loc=posterior_mean, covariance_matrix=posterior_cov
        )
        return posterior.sample((num_samples,))
