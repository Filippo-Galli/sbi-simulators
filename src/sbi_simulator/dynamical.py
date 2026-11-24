"""Shared base class for JAX-accelerated dynamical (SDE/ODE) simulators."""

from typing import Any, Literal, overload, override

import jax
import jax.numpy as jnp
import numpy as np
import torch

from .base import Simulator
from .compositional import sample_two_step


class DynamicalSimulator(Simulator):
    """Base class for continuous-time dynamical trajectory simulators.

    Subclasses provide:
    - ``get_prior()`` and ``posterior_samples(...)``
    - ``_jax_solver_batch`` callable ``(keys, params_jax[, u0_jax]) -> (N, state_dim, T)``
    - Optional ``get_prior_x0()``, ``stepper()``, and ``compute_summary_stats()``
    """

    trajectory_time_axis: int = 2

    def __init__(
        self,
        seed: int = 42,
        dim_parameters: int = 2,
        dim_data: int | None = None,
        *,
        state_dim: int,
        num_steps: int,
        obs_sigma: float = 0.05,
        summary: str | None = None,
        u0: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(
            seed=seed,
            dim_parameters=dim_parameters,
            dim_data=dim_data,
            **kwargs,
        )
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.obs_sigma = float(obs_sigma)
        self.summary = summary
        self.dim_data_raw = int(state_dim * num_steps)
        self.u0 = (
            torch.zeros(state_dim, dtype=torch.float32)
            if u0 is None
            else torch.as_tensor(u0, dtype=torch.float32)
        )
        self._rng_key: jax.Array = jax.random.PRNGKey(self.seed)
        self._jax_solver_batch: Any = None

    @staticmethod
    def _resolve_summary_dim(
        summary: str | None,
        dim_raw: int,
        dim_stats: int,
        dim_two_step: int,
        extra_modes: dict[str, int] | None = None,
    ) -> int:
        if summary is None or summary == "subsample":
            return dim_raw
        if summary in ("stats", "summary_stats"):
            return dim_stats
        if summary == "two_step":
            return dim_two_step
        if extra_modes and summary in extra_modes:
            return extra_modes[summary]
        raise NotImplementedError(f"Unknown summary mode: {summary}")

    def _next_rng_key(self) -> jax.Array:
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    @override
    def reset_rng(self, seed: int | None = None) -> None:
        """Reset internal JAX, NumPy, and Pyro PRNG state for reproducibility."""
        super().reset_rng(seed)
        self._rng_key = jax.random.PRNGKey(self.seed)

    def _solve(
        self,
        parameters: torch.Tensor | np.ndarray,
        u0: torch.Tensor | None = None,
        rng_key: jax.Array | None = None,
    ) -> torch.Tensor:
        """Batched trajectory simulation accelerated via JAX.

        Returns tensor of shape ``(batch_size, state_dim, num_steps)``.
        """
        params_t, _ = self._prepare_parameters(parameters)
        params_jax = jnp.asarray(params_t.detach().cpu().numpy(), dtype=jnp.float32)
        batch_size = params_jax.shape[0]

        if rng_key is None:
            rng_key = self._next_rng_key()
        keys = jax.random.split(rng_key, batch_size)

        if u0 is not None:
            u0_t = torch.as_tensor(u0, dtype=torch.float32)
            if u0_t.ndim == 1:
                u0_t = u0_t.unsqueeze(0)
            if u0_t.shape[0] == 1 and batch_size > 1:
                u0_t = u0_t.expand(batch_size, -1)
            u0_jax = jnp.asarray(u0_t.detach().cpu().numpy(), dtype=jnp.float32)
            res_jax = self._jax_solver_batch(keys, params_jax, u0_jax)
        else:
            res_jax = self._jax_solver_batch(keys, params_jax)

        return torch.from_numpy(np.array(res_jax, copy=True))

    def _format_trajectory_obs(self, x_obs: torch.Tensor | np.ndarray) -> jax.Array:
        """Normalize trajectory observations to JAX array of shape ``(num_steps, state_dim)``."""
        if isinstance(x_obs, np.ndarray):
            x_obs_t = torch.from_numpy(np.array(x_obs, copy=True)).float()
        else:
            x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)

        if x_obs_t.ndim == 2 and x_obs_t.shape[0] == 1:
            x_obs_t = x_obs_t.squeeze(0)

        if x_obs_t.ndim == 1:
            x_obs_2d = x_obs_t.reshape(self.state_dim, -1).T
        elif x_obs_t.ndim == 2 and x_obs_t.shape == (self.state_dim, self.num_steps):
            x_obs_2d = x_obs_t.T
        else:
            x_obs_2d = x_obs_t

        return jnp.asarray(x_obs_2d.detach().cpu().numpy(), dtype=jnp.float32)

    @staticmethod
    def _broadcast_stepper_inputs(
        x: torch.Tensor | np.ndarray,
        parameters: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Normalize and broadcast ``(x, parameters)`` inputs for ``stepper()``."""
        x_t = torch.as_tensor(x, dtype=torch.float32)
        params = torch.as_tensor(parameters, dtype=torch.float32)

        is_1d = x_t.ndim == 1
        if is_1d:
            x_t = x_t.unsqueeze(0)
        if params.ndim == 1:
            params = params.unsqueeze(0)
        if params.shape[0] == 1 and x_t.shape[0] > 1:
            params = params.expand(x_t.shape[0], -1)
        elif x_t.shape[0] == 1 and params.shape[0] > 1:
            x_t = x_t.expand(params.shape[0], -1)
        return x_t, params, is_1d

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
        """Generate two-step transition pairs ``(x_t, x_{t+1})`` for compositional SBI."""
        return sample_two_step(
            self,
            parameters=parameters,
            x0=x0,
            t=t,
            informed=informed,
            return_dict=return_dict,
        )

    @staticmethod
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute summary statistics from ``(batch, state_dim, num_steps)`` trajectories."""
        raise NotImplementedError

    def _apply_summary(self, us_valid: torch.Tensor) -> torch.Tensor:
        """Apply observation model / summary transform to valid trajectories."""
        num_valid = us_valid.shape[0]
        if self.summary == "subsample":
            noisy = us_valid.clone()
            if noisy.shape[-1] > 1:
                noisy[:, :, 1:] = (
                    noisy[:, :, 1:] + torch.randn_like(noisy[:, :, 1:]) * self.obs_sigma
                )
            return noisy.reshape(num_valid, -1)
        if self.summary in ("stats", "summary_stats"):
            return self.compute_summary_stats(us_valid)
        raise ValueError(f"Unknown summary mode: {self.summary}")

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Simulate observations or summary outputs for parameter batch."""
        params, is_1d = self._prepare_parameters(parameters)
        num_samples = params.shape[0]

        if self.summary == "two_step":
            pairs = self.simulate_two_step(params, return_dict=False)
            out = pairs.reshape(num_samples, 2 * self.state_dim)
            return out.squeeze(0) if is_1d else out

        us = self._solve(params).float()  # (num_samples, state_dim, num_steps)

        if self.summary is None:
            out = us.reshape(num_samples, -1)
            return out.squeeze(0) if is_1d else out

        has_nan = torch.isnan(us.reshape(num_samples, -1)).any(dim=1)
        if not has_nan.any():
            out = self._apply_summary(us)
            return out.squeeze(0) if is_1d else out

        data = torch.full(
            (num_samples, self.dim_data), float("nan"), dtype=torch.float32
        )
        valid_mask = ~has_nan
        if valid_mask.any():
            data[valid_mask] = self._apply_summary(us[valid_mask])
        return data.squeeze(0) if is_1d else data

    def _finalize_mcmc_samples(
        self,
        samples_jax: jax.Array,
        num_samples: int,
        *,
        shuffle: bool = True,
    ) -> torch.Tensor:
        """Convert JAX MCMC chains to a shuffled ``(num_samples, dim_parameters)`` tensor."""
        samples_np = np.array(samples_jax, copy=True)
        if shuffle:
            self._np_rng.shuffle(samples_np)
        return torch.from_numpy(samples_np[:num_samples])
