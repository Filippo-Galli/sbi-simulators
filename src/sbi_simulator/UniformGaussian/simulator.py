"""Simulator for the Uniform-Gaussian model.

Reference:
    Samaroo, K., Shen, Y., Ryder, R. J., & Broderick, T. (2026).
    "Quit While You're Ahead: Sequential Training Can Harm Simulation-Based Inference."
    Symposium on Probabilistic Machine Learning (ProbML 2026).
    https://openreview.net/forum?id=2T0yUSGwFa
"""

from __future__ import annotations

import math
from typing import override

import numpy as np
import pyro.distributions as pdist
import torch
from scipy import stats

from ..base import Simulator
from ..sampling import rejection_sample_box


class UniformGaussian(Simulator):
    """Uniform-Gaussian simulator as introduced in Samaroo et al. (2026).

    Model specification:
        - Parameter prior: ``theta ~ Uniform([-L_d, L_d]^d)``
        - Data likelihood: ``x | theta ~ N(theta, sigma^2 * I_d)``
        - Observed data: ``x_obs = 0`` (by default)
        - Ground truth posterior: ``theta | x_obs ~ N_{[-L_d, L_d]^d}(x_mean, (sigma^2 / N) * I_d)``
    """

    def __init__(
        self,
        dim_parameters: int = 10,
        seed: int = 42,
        sigma: float | None = None,
        dim_data: int | None = None,
        alpha: float = 0.99,
        L: float | None = None,
        preset: str = "default",
        **kwargs,
    ):
        preset_key = preset.lower()
        if preset_key not in ("default", "sbibm"):
            raise ValueError(
                f"Unknown preset '{preset}'. Supported presets are 'default' and 'sbibm'."
            )
        self.preset = preset_key
        if sigma is None:
            self.sigma = math.sqrt(0.1) if self.preset == "sbibm" else 1.0
        else:
            self.sigma = float(sigma)
        self.alpha = float(alpha)

        if dim_data is None:
            dim_data = dim_parameters

        super().__init__(
            seed=seed, dim_parameters=dim_parameters, dim_data=dim_data, **kwargs
        )

        if L is not None:
            self.L = float(L)
        elif self.preset == "sbibm":
            self.L = 1.0
        else:
            self.L = float(self.compute_prior_bound(self.dim_parameters, self.alpha))

    @staticmethod
    def compute_prior_bound(dim: int, alpha: float = 0.99) -> float:
        """Compute the prior bound ``L_d = sqrt(F_{chi^2_d}^{-1}(alpha))``."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        return float(np.sqrt(stats.chi2.ppf(alpha, df=dim)))

    @property
    def default_x_obs(self) -> torch.Tensor:
        """Standard conditioning point ``x_obs = 0``."""
        return torch.zeros(self.dim_data, dtype=torch.float32)

    @property
    def default_theta_true(self) -> torch.Tensor:
        """Ground truth ``theta = 0`` associated with ``default_x_obs``."""
        return torch.zeros(self.dim_parameters, dtype=torch.float32)

    @override
    def get_prior(self) -> pdist.Distribution:
        """Return the uniform prior ``Uniform([-L_d, L_d]^d)``."""
        low = -self.L * torch.ones(self.dim_parameters, dtype=torch.float32)
        high = self.L * torch.ones(self.dim_parameters, dtype=torch.float32)
        return pdist.Uniform(low, high).to_event(1)

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Simulate synthetic data ``x = parameters + sigma * noise``, ``noise ~ N(0, I_d)``."""
        params, is_1d = self._prepare_parameters(parameters)
        if params.shape[-1] != self.dim_parameters:
            raise ValueError(
                f"Expected {self.dim_parameters} parameters on last dimension, "
                f"got {params.shape[-1]}."
            )
        out = params + self.sigma * torch.randn_like(params)
        return out.squeeze(0) if is_1d else out

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Draw exact samples from ``theta | x_obs ~ N_{[-L_d, L_d]^d}(x_mean, (sigma^2 / N) * I_d)``."""
        if x_obs is None:
            x_obs_t = self.default_x_obs.unsqueeze(0)
        else:
            x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
            if x_obs_t.ndim == 1:
                x_obs_t = x_obs_t.unsqueeze(0)

        n_obs, d = x_obs_t.shape
        if d != self.dim_parameters:
            raise ValueError(
                f"Expected x_obs with {self.dim_parameters} elements per observation, got {d}."
            )

        L = self.L
        effective_sigma = self.sigma / math.sqrt(n_obs)
        x_mean = x_obs_t.mean(dim=0)

        def propose(batch_size: int) -> torch.Tensor:
            return x_mean + effective_sigma * torch.randn(
                (batch_size, d), dtype=torch.float32, device=x_mean.device
            )

        def fallback(remaining: int) -> torch.Tensor:
            x_np = x_mean.cpu().numpy()
            a = (-L - x_np) / effective_sigma
            b = (L - x_np) / effective_sigma
            rvs = stats.truncnorm.rvs(
                a,
                b,
                loc=x_np,
                scale=effective_sigma,
                size=(remaining, d),
                random_state=self._np_rng,
            )
            return torch.as_tensor(rvs, dtype=torch.float32)

        return rejection_sample_box(
            propose,
            num_samples,
            low=-L,
            high=L,
            max_attempts=15,
            fallback_fn=fallback,
        )
