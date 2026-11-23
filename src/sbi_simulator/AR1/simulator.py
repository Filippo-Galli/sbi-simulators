"""Simulator for the First-Order Autoregressive (AR(1)) process."""

import warnings
from typing import Literal, overload, override

import numpy as np
import pyro.distributions as pdist
import torch

from ..base import Simulator
from ..compositional import sample_two_step
from ..summary_stats import trajectory_stats_interleaved


class AR1(Simulator):
    r"""Simulates a first-order autoregressive AR(1) process with Gaussian white noise.

    .. math::
        y_0 \sim \mathcal{N}(0, \sigma_0^2) \\
        y_t = \phi y_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)

    Parameters
    ----------
    dim_parameters : int, optional
        Dimension of the parameters (autoregressive coefficient ``phi``), must be 1.
    seed : int, optional
        Random seed, by default 42.
    dim_data : int, optional
        Timeseries length ``T`` when ``summary=None``, by default 100.
    sigma : float, optional
        Standard deviation of the white noise, by default 1.0.
    summary : str or None, optional
        Output mode: ``None`` (full trajectory), ``"stats"`` / ``"summary_stats"``
        (4 stats: mean, log-variance, lag-1 and lag-2 autocorrelation), or
        ``"two_step"`` (flattened transition pair ``[y_t, y_{t+1}]``).
    """

    trajectory_time_axis: int = 2

    def __init__(
        self,
        dim_parameters: int = 1,
        seed: int = 42,
        dim_data: int = 100,
        sigma: float = 1.0,
        summary: str | None = None,
        **kwargs,
    ):
        if dim_parameters != 1:
            raise ValueError(
                f"AR1 requires dim_parameters=1 (phi), got {dim_parameters}."
            )

        self.timeseries_length = int(dim_data)
        self.sigma = float(sigma)
        self.prior_std = 0.2
        self.x0_std = 0.4
        self.summary = summary

        if summary is None:
            resolved_dim_data = self.timeseries_length
        elif summary in ("stats", "summary_stats"):
            resolved_dim_data = 4
        elif summary == "two_step":
            resolved_dim_data = 2
        else:
            raise NotImplementedError(f"Unknown summary mode: {summary}")

        super().__init__(
            seed=seed,
            dim_parameters=dim_parameters,
            dim_data=resolved_dim_data,
            **kwargs,
        )
        self.u0 = torch.zeros(1, dtype=torch.float32)

    @override
    def get_prior(self) -> pdist.Distribution:
        """Prior over autoregressive coefficient ``phi ~ N(0, 0.2^2)``."""
        return pdist.Normal(
            torch.zeros(self.dim_parameters),
            self.prior_std * torch.ones(self.dim_parameters),
        ).to_event(1)

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial condition ``y_0 ~ N(0, 0.4^2)``."""
        return pdist.Normal(torch.zeros(1), self.x0_std * torch.ones(1)).to_event(1)

    def stepper(
        self,
        y: torch.Tensor | np.ndarray,
        phi: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Simulate one AR(1) transition step ``y_t = phi * y_{t-1} + eps``."""
        y_t = torch.as_tensor(y, dtype=torch.float32)
        phi_t = torch.as_tensor(phi, dtype=torch.float32, device=y_t.device)
        is_1d = y_t.ndim <= 1
        if y_t.ndim == 0:
            y_t = y_t.reshape(1, 1)
        elif y_t.ndim == 1:
            y_t = y_t.unsqueeze(0)
        if phi_t.ndim == 0:
            phi_t = phi_t.reshape(1, 1)
        elif phi_t.ndim == 1:
            phi_t = phi_t.unsqueeze(0)

        if phi_t.shape[0] == 1 and y_t.shape[0] > 1:
            phi_t = phi_t.expand(y_t.shape[0], -1)
        elif y_t.shape[0] == 1 and phi_t.shape[0] > 1:
            y_t = y_t.expand(phi_t.shape[0], -1)

        next_y = phi_t[:, :1] * y_t + self.sigma * torch.randn_like(y_t)
        return next_y.squeeze(0) if is_1d else next_y

    @staticmethod
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute 4 summary stats ``[mean, log_var, ac1, ac2]`` from ``(B, 1, T)``."""
        return trajectory_stats_interleaved(us)

    def _solve(
        self,
        parameters: torch.Tensor | np.ndarray,
        u0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Simulate AR(1) trajectories of shape ``(batch_size, 1, timeseries_length)``."""
        params, _ = self._prepare_parameters(parameters)
        batch_size = params.shape[0]
        device = params.device

        out = torch.empty(
            (batch_size, 1, self.timeseries_length), dtype=torch.float32, device=device
        )

        if u0 is None:
            out[:, :, 0] = self.get_prior_x0().sample((batch_size,)).to(device)
        else:
            u0_t = torch.as_tensor(u0, dtype=torch.float32, device=device).reshape(
                -1, 1
            )
            if u0_t.shape[0] == 1 and batch_size > 1:
                u0_t = u0_t.expand(batch_size, -1)
            out[:, :, 0] = u0_t

        for t in range(1, self.timeseries_length):
            out[:, :, t] = self.stepper(out[:, :, t - 1], params)

        return out

    @overload
    def simulate_two_step(
        self,
        parameters: torch.Tensor | np.ndarray,
        x0: torch.Tensor | np.ndarray | None = None,
        t: int | torch.Tensor | None = None,
        informed: bool = True,
        return_dict: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def simulate_two_step(
        self,
        parameters: torch.Tensor | np.ndarray,
        x0: torch.Tensor | np.ndarray | None = None,
        t: int | torch.Tensor | None = None,
        informed: bool = True,
        return_dict: Literal[True] = ...,
    ) -> dict[str, torch.Tensor]: ...

    @overload
    def simulate_two_step(
        self,
        parameters: torch.Tensor | np.ndarray,
        x0: torch.Tensor | np.ndarray | None = None,
        t: int | torch.Tensor | None = None,
        informed: bool = True,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]: ...

    def simulate_two_step(
        self,
        parameters: torch.Tensor | np.ndarray,
        x0: torch.Tensor | np.ndarray | None = None,
        t: int | torch.Tensor | None = None,
        informed: bool = True,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Generate two-step transition pairs ``(y_t, y_{t+1})`` for compositional SBI."""
        return sample_two_step(
            self,
            parameters=parameters,
            x0=x0,
            t=t,
            informed=informed,
            return_dict=return_dict,
        )

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Simulate AR(1) trajectories, summary stats, or two-step pairs."""
        params, is_1d = self._prepare_parameters(parameters)
        num_samples = params.shape[0]

        if self.summary == "two_step":
            pairs = self.simulate_two_step(params, return_dict=False)
            out = pairs.reshape(num_samples, 2)
            return out.squeeze(0) if is_1d else out

        us = self._solve(params)

        if self.summary is None:
            out = us.squeeze(1)
            return out.squeeze(0) if is_1d else out

        if self.summary in ("stats", "summary_stats"):
            out = self.compute_summary_stats(us)
            return out.squeeze(0) if is_1d else out

        raise ValueError(f"Unknown summary mode: {self.summary}")

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the exact analytic posterior ``phi | y`` under ``phi ~ N(0, prior_std^2)``.

        Linear regression likelihood: ``y_t | y_{t-1}, phi ~ N(phi * y_{t-1}, sigma^2)``.
        Conjugate posterior: ``phi | y ~ N(mu_post, sigma_post^2)``.
        """
        if x_obs is None:
            warnings.warn("x_obs is None, using zeros as placeholder", stacklevel=2)
            x_obs = torch.zeros(self.timeseries_length, dtype=torch.float32)

        x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_t.ndim == 1:
            x_obs_t = x_obs_t.unsqueeze(0)

        y_prev = x_obs_t[:, :-1].flatten()
        y_curr = x_obs_t[:, 1:].flatten()

        var_noise = self.sigma**2
        prec_prior = 1.0 / (self.prior_std**2)
        prec_lik = (y_prev**2).sum() / var_noise
        prec_post = prec_prior + prec_lik
        var_post = 1.0 / prec_post
        mean_post = var_post * (y_curr * y_prev).sum() / var_noise

        posterior = torch.distributions.Normal(mean_post, torch.sqrt(var_post))
        return posterior.sample((num_samples, self.dim_parameters))
