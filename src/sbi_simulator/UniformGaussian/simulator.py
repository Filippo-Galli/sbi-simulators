"""Simulator for the Uniform-Gaussian model.

Reference:
    Samaroo, K., Shen, Y., Ryder, R. J., & Broderick, T. (2026).
    "Quit While You're Ahead: Sequential Training Can Harm Simulation-Based Inference."
    Symposium on Probabilistic Machine Learning (ProbML 2026).
    https://openreview.net/forum?id=2T0yUSGwFa
"""

from __future__ import annotations

from typing import override

import numpy as np
import pyro.distributions as pdist
import torch
from scipy import stats

from ..Simulator import Simulator


class UniformGaussian(Simulator):
    """Uniform-Gaussian simulator as introduced in Samaroo et al. (2026).

    Model specification:
        - Parameter prior: theta ~ Uniform([-L_d, L_d]^d)
        - Data likelihood: x | theta ~ N(theta, sigma^2 * I_d)
        - Observed data: x_obs = 0 (by default)
        - Ground truth posterior:
            theta | {x_obs = 0} ~ N_{[-L_d, L_d]^d}(0, sigma^2 * I_d)

    Prior width L_d:
        Defined as L_d := sqrt(F_{chi^2_d}^{-1}(alpha)) where alpha = 0.99 by default,
        ensuring that the hypersphere S_{L_d} := {theta: ||theta||_2 <= L_d} is a 99%
        credible region of the untruncated posterior N(0, I_d).

    """

    def __init__(
        self,
        dim_parameters: int = 10,
        seed: int = 42,
        sigma: float = 1.0,
        dim_data: int | None = None,
        alpha: float = 0.99,
        L: float | None = None,
    ):
        self.sigma = float(sigma)
        self.alpha = float(alpha)

        if dim_data is None:
            dim_data = dim_parameters

        super().__init__(seed=seed, dim_parameters=dim_parameters, dim_data=dim_data)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if L is not None:
            self.L = float(L)
        else:
            self.L = float(self.compute_prior_bound(self.dim_parameters, self.alpha))

    @staticmethod
    def compute_prior_bound(dim: int, alpha: float = 0.99) -> float:
        """Compute the prior bound L_d = sqrt(F_{chi^2_d}^{-1}(alpha)).

        Args:
            dim: Dimension of the parameter space.
            alpha: Confidence level for the prior bound.

        Returns:
            The prior bound L_d.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        return float(np.sqrt(stats.chi2.ppf(alpha, df=dim)))

    @property
    def default_x_obs(self) -> torch.Tensor:
        """Standard conditioning point x_obs = 0 as studied in the paper."""
        return torch.zeros(self.dim_data, dtype=torch.float32)

    @property
    def default_theta_true(self) -> torch.Tensor:
        """Ground truth theta = 0 associated with default x_obs = 0."""
        return torch.zeros(self.dim_parameters, dtype=torch.float32)

    @override
    def get_prior(self) -> pdist.Distribution:
        """Return the uniform prior Uniform([-L_d, L_d]^d)."""
        low = -self.L * torch.ones(self.dim_parameters, dtype=torch.float32)
        high = self.L * torch.ones(self.dim_parameters, dtype=torch.float32)
        return pdist.Uniform(low, high).to_event(1)

    def sample_prior(self, num_samples: int = 1000) -> torch.Tensor:
        """Draw samples from the prior distribution theta ~ Uniform([-L_d, L_d]^d)."""
        return self.get_prior().sample((num_samples,))

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Simulate synthetic data x = parameters + sigma * noise, noise ~ N(0, I_d).

        Args:
            parameters: 1D tensor/array of shape (d,) or 2D batched of shape (N, d).

        Returns:
            Simulated data with matching shape.
        """
        params = torch.as_tensor(parameters, dtype=torch.float32)
        if params.numel() == 0:
            raise ValueError("Expected at least one parameter.")

        if params.ndim == 1:
            if params.shape[0] != self.dim_parameters:
                raise ValueError(
                    f"Expected {self.dim_parameters} parameters, got {params.shape[0]}."
                )
        elif params.ndim >= 2 and params.shape[-1] != self.dim_parameters:
            raise ValueError(
                f"Expected {self.dim_parameters} parameters on last dimension, "
                f"got {params.shape[-1]}."
            )

        return params + self.sigma * torch.randn_like(params)

    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Draw exact samples from the ground truth posterior:
        theta | x_obs ~ N_{[-L_d, L_d]^d}(x_mean, (sigma^2 / N) * I_d).

        Args:
            x_obs: Observation to condition on (1D or NxDim). Defaults to 0 (the paper's benchmark setting).
            num_samples: Number of posterior samples to draw.

        Returns:
            torch.Tensor of shape (num_samples, dim_parameters).
        """
        if (
            x_obs is None
            and "num_samples" in kwargs
            and isinstance(num_samples, (torch.Tensor, np.ndarray))
        ):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)
        if x_obs is None:
            x_obs_t = self.default_x_obs.unsqueeze(0)
        else:
            x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
            if x_obs_t.ndim == 1:
                x_obs_t = x_obs_t.unsqueeze(0)

        n_obs = x_obs_t.shape[0]
        d = x_obs_t.shape[-1]
        if d != self.dim_parameters:
            raise ValueError(
                f"Expected x_obs with {self.dim_parameters} elements per observation, got {d}."
            )

        L = self.L
        effective_sigma = self.sigma / np.sqrt(n_obs)
        x_mean = x_obs_t.mean(dim=0)

        # Rejection sampling:
        samples_list: list[torch.Tensor] = []
        collected = 0
        max_rejection_attempts = 15
        attempts = 0

        while collected < num_samples and attempts < max_rejection_attempts:
            attempts += 1
            needed = num_samples - collected
            batch_size = max(needed * 2, 1000)
            candidates = x_mean + effective_sigma * torch.randn(
                batch_size, d, dtype=torch.float32
            )
            mask = torch.all((candidates >= -L) & (candidates <= L), dim=-1)
            accepted = candidates[mask]
            if accepted.numel() > 0:
                n_take = min(accepted.shape[0], needed)
                samples_list.append(accepted[:n_take])
                collected += n_take

        if collected >= num_samples:
            return torch.cat(samples_list, dim=0)

        # Fallback to independent coordinate-wise truncated normal if rejection rate is low
        # (e.g. if x_mean is located near or outside the boundary).
        remaining = num_samples - collected
        x_np = x_mean.cpu().numpy()
        a = (-L - x_np) / effective_sigma
        b = (L - x_np) / effective_sigma
        fallback_samples = stats.truncnorm.rvs(
            a, b, loc=x_np, scale=effective_sigma, size=(remaining, d)
        )
        fallback_t = torch.as_tensor(fallback_samples, dtype=torch.float32)
        if samples_list:
            return torch.cat([torch.cat(samples_list, dim=0), fallback_t], dim=0)
        return fallback_t
