"""Simulator for the First-Order Autoregressive (AR(1)) process."""

import math
import warnings
from typing import override

import pyro.distributions as pdist
import torch

from ..Simulator import Simulator


class AR1(Simulator):
    r"""Simulates a first-order autoregressive AR(1) process with Gaussian white noise.

    .. math::
        y_0 \sim \mathcal{N}(0, \sigma_0^2) \\
        y_t = \phi y_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)

    Parameters
    ----------
    dim_parameters : int, optional
        Dimension of the parameters (autoregressive coefficient \phi), by default 1.
    seed : int, optional
        Random seed, by default 42.
    dim_data : int, optional
        Dimension of the data (timeseries length T), by default 100.
    sigma : float, optional
        Standard deviation of the white noise, by default 1.0.
    """

    def __init__(
        self,
        dim_parameters: int = 1,
        seed: int = 42,
        dim_data: int = 100,
        sigma: float = 1.0,
    ):
        self.timeseries_length = dim_data
        self.sigma = float(sigma)
        self.prior_std = 0.2
        self.x0_std = 0.4
        super().__init__(seed=seed, dim_parameters=dim_parameters, dim_data=dim_data)

    @override
    def get_prior(self) -> pdist.Distribution:
        """Prior over autoregressive coefficient phi ~ N(0, 0.2^2)."""
        return pdist.Normal(
            torch.zeros(self.dim_parameters),
            self.prior_std * torch.ones(self.dim_parameters),
        ).to_event(1)

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial condition y_0 ~ N(0, 0.4^2)."""
        return pdist.Normal(torch.zeros(1), self.x0_std * torch.ones(1)).to_event(1)

    def stepper(self, y: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Simulate AR(1) trajectory step."""
        return phi * y + self.sigma * torch.randn_like(y)

    @override
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        """Simulate AR(1) trajectories.

        Parameters
        ----------
        parameters : torch.Tensor
            Autoregressive coefficient phi, shape (dim_parameters,) or (batch_size, dim_parameters).

        Returns
        -------
        torch.Tensor
            Simulated timeseries of shape (dim_data,) or (batch_size, dim_data).
        """
        params = torch.as_tensor(parameters, dtype=torch.float32)
        if params.numel() == 0:
            raise ValueError("Expected at least one parameter.")

        is_1d = params.ndim == 1
        if is_1d:
            params = params.unsqueeze(0)

        batch_size = params.shape[0]
        device = params.device

        # Pre-allocate trajectory
        out = torch.empty(
            (batch_size, self.timeseries_length), dtype=torch.float32, device=device
        )

        out[:, 0] = self.get_prior_x0().sample((batch_size,)).squeeze(-1)

        phi = params[:, 0]
        for t in range(1, self.timeseries_length):
            out[:, t] = self.stepper(out[:, t - 1], phi)

        return out.squeeze(0) if is_1d else out

    def posterior_samples(
        self,
        x_obs: torch.Tensor | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the exact analytic posterior phi | y under prior phi ~ N(0, prior_std^2).

        Linear regression likelihood: y_t | y_{t-1}, phi ~ N(phi * y_{t-1}, sigma^2).
        Conjugate posterior: phi | y ~ N(mu_post, sigma_post^2).
        """
        if (
            x_obs is None
            and "num_samples" in kwargs
            and isinstance(num_samples, torch.Tensor)
        ):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)
        if x_obs is None:
            warnings.warn("x_obs is None, using zeros as placeholder")
            x_obs = torch.zeros(self.dim_data, dtype=torch.float32)

        x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_t.ndim == 1:
            x_obs_t = x_obs_t.unsqueeze(0)

        # y_{t-1} and y_t across all observed series
        y_prev = x_obs_t[:, :-1].flatten()
        y_curr = x_obs_t[:, 1:].flatten()

        var_noise = self.sigma**2
        prec_prior = 1.0 / (self.prior_std**2)
        prec_lik = (y_prev**2).sum() / var_noise
        prec_post = prec_prior + prec_lik
        var_post = 1.0 / prec_post
        mean_post = var_post * (y_curr * y_prev).sum() / var_noise

        posterior = torch.distributions.Normal(mean_post, math.sqrt(var_post))
        samples = posterior.sample((num_samples, self.dim_parameters))
        return samples
