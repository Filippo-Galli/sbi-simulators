"""Simulator for the Gaussian-Gaussian model.
reference: 'Compositional Score Modeling for Simulation-Based Inference'
"""

import math
from typing import override

import numpy as np
import pyro.distributions as pdist
import torch

from ..base import Simulator


class GaussianGaussian(Simulator):
    """Simulator for the Gaussian-Gaussian / Gaussian Linear model.

    Presets:
    --------
    - ``"default"``:
      Configuration from *Compositional Score Modeling for Simulation-Based Inference*:
      Prior ``N(0, I_d)``, Likelihood ``N(x | theta, diag_sigma)`` with variances
      linearly spaced in ``[0.6, 1.4]``.
    - ``"sbibm"`` (Lueckmann et al., 2021, arXiv:2101.04653, Task ``'gaussian_linear'``):
      Prior ``N(0, 0.1 * I_d)``, Likelihood ``N(x | theta, 0.1 * I_d)``.
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
        if dim == 1:
            return torch.tensor([0.6], dtype=torch.float32)
        return 0.6 + torch.arange(dim, dtype=torch.float32) * (1.4 - 0.6) / (dim - 1)

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        params, is_1d = self._prepare_parameters(parameters)
        dim = params.shape[-1]
        std = torch.sqrt(self._diag_sigma(dim).to(params.device))
        out = params + torch.randn_like(params) * std
        return out.squeeze(0) if is_1d else out

    @override
    def get_prior(self) -> pdist.Distribution:
        std = math.sqrt(0.1) if self.preset == "sbibm" else 1.0
        return pdist.Normal(
            torch.zeros(self.dim_parameters),
            std * torch.ones(self.dim_parameters),
        ).to_event(1)

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the exact closed-form diagonal Gaussian posterior in ``O(D)`` time.

        Supports single observation ``(D,)`` / ``(1, D)`` or multiple i.i.d.
        observations ``(N, D)``:
            ``var_post = 1 / (1 / prior_var + N / diag_sigma)``
            ``mean_post = var_post * (sum_i x_i / diag_sigma)``
        """
        if x_obs is None:
            x_obs_t = torch.zeros((1, self.dim_data), dtype=torch.float32)
        else:
            x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
            if x_obs_t.ndim == 1:
                x_obs_t = x_obs_t.unsqueeze(0)

        n_obs, dim = x_obs_t.shape
        diag_sigma = self._diag_sigma(dim).to(x_obs_t.device)
        prior_var = 0.1 if self.preset == "sbibm" else 1.0

        post_var = 1.0 / (1.0 / prior_var + n_obs / diag_sigma)
        post_mean = post_var * (x_obs_t.sum(dim=0) / diag_sigma)
        return post_mean + torch.sqrt(post_var) * torch.randn(
            (num_samples, dim), dtype=torch.float32, device=x_obs_t.device
        )
