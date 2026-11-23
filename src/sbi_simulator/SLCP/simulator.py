"""Simulator for the Simple Likelihood Complex Posterior (SLCP) benchmark model."""

import functools
import math
from typing import Literal, overload, override

import jax
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pdist
import torch

from ..base import Simulator
from ..compositional import sample_two_step
from ..mcmc import build_batched_ll, build_mh_mcmc_sampler


@functools.lru_cache(maxsize=4)
def _get_slcp_jax_bundle():
    """Build and cache JIT-compiled SLCP log-likelihood, batched vmap, and MCMC sampler."""

    def slcp_log_likelihood(theta, x_obs):
        in_bounds = jnp.all((theta >= -3.0) & (theta <= 3.0))

        m = theta[:2]
        s1 = theta[2] ** 2
        s2 = theta[3] ** 2
        rho = jnp.tanh(theta[4])

        eps = 1e-6
        var1 = s1**2 + eps
        var2 = s2**2 + eps
        cov = rho * s1 * s2

        det = jnp.maximum(var1 * var2 - cov**2, 1e-12)
        inv00 = var2 / det
        inv11 = var1 / det
        inv01 = -cov / det

        diff = x_obs - m
        quad = (
            diff[:, 0] ** 2 * inv00
            + 2.0 * diff[:, 0] * diff[:, 1] * inv01
            + diff[:, 1] ** 2 * inv11
        )
        log_2pi = jnp.log(2.0 * jnp.pi)
        log_det = jnp.log(det)
        ll = jnp.sum(-0.5 * (quad + 2.0 * log_2pi + log_det))

        return jnp.where(in_bounds & jnp.isfinite(ll), ll, -jnp.inf)

    ll_fn = jax.jit(slcp_log_likelihood)
    v_ll_fn = build_batched_ll(ll_fn, stochastic=False)
    jax_mcmc = build_mh_mcmc_sampler(
        ll_fn,
        dim_parameters=5,
        prior_type="uniform",
        prior_low=-3.0,
        prior_high=3.0,
        init_strategy="sir_resample",
    )
    return v_ll_fn, jax_mcmc


class SLCP(Simulator):
    r"""Simple Likelihood Complex Posterior (SLCP) benchmark simulator.

    Reference:
        Papamakarios et al. (2019), "Sequential Neural Likelihood".
        Lueckmann et al. (2021), "Benchmarking Simulation-Based Inference".

    Parameters:
        ``theta in R^5 ~ Uniform(-3, 3)^5``

    Likelihood:
        Generates ``num_data`` independent 2D observations
        ``y_1, ..., y_K ~ N(m(theta), Sigma(theta))`` where:
            ``m(theta) = [theta_0, theta_1]^T``
            ``s_1 = theta_2^2, s_2 = theta_3^2, rho = tanh(theta_4)``
            ``Sigma(theta) = [[s_1^2 + eps, rho * s_1 * s_2], [rho * s_1 * s_2, s_2^2 + eps]]``
    """

    trajectory_time_axis: int = 1

    def __init__(
        self,
        dim_parameters: int = 5,
        seed: int = 42,
        num_data: int = 4,
        summary: str | None = None,
        **kwargs,
    ):
        if dim_parameters != 5:
            raise ValueError(
                f"SLCP requires dim_parameters=5 (theta_0 to theta_4), got {dim_parameters}."
            )

        self.num_data = num_data
        self.dim_data_raw = int(2 * num_data)

        if summary is None or summary == "subsample":
            dim_data = self.dim_data_raw
        elif summary in ("stats", "summary_stats"):
            dim_data = 5  # mean_x, mean_y, logvar_x, logvar_y, corr_xy
        elif summary == "two_step":
            dim_data = 4  # [x_t, x_{t+1}] in R^4
        else:
            raise NotImplementedError(f"Unknown summary mode: {summary}")
        self.summary = summary

        super().__init__(seed, dim_parameters, dim_data, **kwargs)

        self.prior_low = -3.0
        self.prior_high = 3.0
        self.prior_dist = pdist.Uniform(
            low=torch.full((5,), self.prior_low, dtype=torch.float32),
            high=torch.full((5,), self.prior_high, dtype=torch.float32),
        ).to_event(1)

        self.u0 = torch.zeros(2, dtype=torch.float32)
        self._rng_key: jax.Array = jax.random.PRNGKey(self.seed)
        self._v_ll_fn, self._jax_mcmc = _get_slcp_jax_bundle()

    @override
    def reset_rng(self, seed: int | None = None) -> None:
        super().reset_rng(seed)
        self._rng_key = jax.random.PRNGKey(self.seed)

    def _next_rng_key(self) -> jax.Array:
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial point ``x0`` in ``R^2``."""
        return pdist.Uniform(
            low=torch.tensor([-15.0, -15.0], dtype=torch.float32),
            high=torch.tensor([15.0, 15.0], dtype=torch.float32),
        ).to_event(1)

    @staticmethod
    def _gaussian_params(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = params.shape[0]
        m = params[:, :2]
        s1 = params[:, 2] ** 2
        s2 = params[:, 3] ** 2
        rho = torch.tanh(params[:, 4])

        eps = 1e-6
        var1 = s1**2 + eps
        var2 = s2**2 + eps
        cov = rho * s1 * s2

        S = torch.empty((batch_size, 2, 2), dtype=torch.float32, device=params.device)
        S[:, 0, 0] = var1
        S[:, 0, 1] = cov
        S[:, 1, 0] = cov
        S[:, 1, 1] = var2
        return m, S

    def stepper(
        self,
        x: torch.Tensor | np.ndarray,
        parameters: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Generate one i.i.d. transition step from ``parameters``."""
        x_t = torch.as_tensor(x, dtype=torch.float32)
        params = torch.as_tensor(parameters, dtype=torch.float32)
        is_1d = params.ndim == 1 and x_t.ndim == 1
        if params.ndim == 1:
            params = params.unsqueeze(0)
        if x_t.ndim > 1 and params.shape[0] == 1 and x_t.shape[0] > 1:
            params = params.expand(x_t.shape[0], -1)

        m, S = self._gaussian_params(params)
        next_x = pdist.MultivariateNormal(m, S).sample()
        return next_x.squeeze(0) if is_1d else next_x

    @staticmethod
    def compute_summary_stats(xs: torch.Tensor) -> torch.Tensor:
        """Compute 5 summary statistics from ``(num_samples, K, 2)`` observations."""
        is_2d = xs.ndim == 2
        if is_2d:
            xs = xs.unsqueeze(0)

        eps = 1e-6
        mean_0 = xs[:, :, 0].mean(dim=-1, keepdim=True)
        mean_1 = xs[:, :, 1].mean(dim=-1, keepdim=True)

        d0 = xs[:, :, 0] - mean_0
        d1 = xs[:, :, 1] - mean_1

        var_0 = (d0**2).mean(dim=-1, keepdim=True)
        var_1 = (d1**2).mean(dim=-1, keepdim=True)

        log_var_0 = torch.log(var_0.clamp_min(eps))
        log_var_1 = torch.log(var_1.clamp_min(eps))

        denom = torch.sqrt(var_0.clamp_min(eps) * var_1.clamp_min(eps))
        corr = (d0 * d1).mean(dim=-1, keepdim=True) / denom
        corr = torch.nan_to_num(corr.clamp(-1.0, 1.0), nan=0.0)

        stats = torch.cat([mean_0, mean_1, log_var_0, log_var_1, corr], dim=-1)
        return stats.squeeze(0) if is_2d else stats

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

    def _solve(
        self, parameters: torch.Tensor, u0: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Simulate ``num_data`` 2D observations given parameters -> ``(B, num_data, 2)``."""
        _ = u0
        params, is_1d = self._prepare_parameters(parameters)
        batch_size = params.shape[0]
        m, S = self._gaussian_params(params)
        dist = pdist.MultivariateNormal(m.unsqueeze(1), S.unsqueeze(1)).expand(
            (batch_size, self.num_data)
        )
        data = dist.sample()
        return data.squeeze(0) if is_1d else data

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        params, is_1d = self._prepare_parameters(parameters)
        num_samples = params.shape[0]

        if self.summary == "two_step":
            pairs = self.simulate_two_step(params, return_dict=False)
            pairs_flat = pairs.reshape(num_samples, 4)
            return pairs_flat.squeeze(0) if is_1d else pairs_flat

        xs = self._solve(params)

        if self.summary is None or self.summary == "subsample":
            xs_flat = xs.reshape(num_samples, self.dim_data_raw)
            return xs_flat.squeeze(0) if is_1d else xs_flat

        if self.summary in ("stats", "summary_stats"):
            stats = self.compute_summary_stats(xs)
            return stats.squeeze(0) if is_1d else stats

        raise ValueError(f"Unknown summary mode: {self.summary}")

    def log_likelihood(
        self,
        parameters: torch.Tensor | np.ndarray,
        x_obs: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Evaluate exact analytical log-likelihood of ``x_obs`` under ``parameters``."""
        params, _ = self._prepare_parameters(parameters)
        x_obs_2d = (
            torch.as_tensor(x_obs, dtype=torch.float32)
            .flatten()
            .reshape(self.num_data, 2)
        )
        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy(), dtype=jnp.float32)
        params_jax = jnp.asarray(params.detach().cpu().numpy(), dtype=jnp.float32)

        lls_jax = self._v_ll_fn(params_jax, x_jax)
        return torch.from_numpy(np.array(lls_jax, copy=True))

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        n_chains: int = 20,
        burn_in: int = 500,
        thin: int = 4,
        step_size: float = 0.08,
        n_sir: int = 5000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the true multimodal posterior ``p(theta | x_obs)`` via JAX MCMC."""
        if x_obs is None:
            # Benchmark observation 1 from sbibm
            x_obs = torch.tensor(
                [
                    2.3718784,
                    0.49947417,
                    9.931435,
                    1.7136912,
                    -10.436423,
                    -1.9067793,
                    -1.2343777,
                    -0.09735,
                ],
                dtype=torch.float32,
            )

        x_obs_2d = (
            torch.as_tensor(x_obs, dtype=torch.float32)
            .flatten()
            .reshape(self.num_data, 2)
        )
        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy(), dtype=jnp.float32)

        num_samples_per_chain = math.ceil(num_samples / n_chains)
        samples_jax = self._jax_mcmc(
            x_jax,
            self._next_rng_key(),
            n_chains=n_chains,
            num_samples_per_chain=num_samples_per_chain,
            burn_in=burn_in,
            inner_thin=thin,
            step_size=step_size,
            n_sir=n_sir,
        )

        samples_np = np.array(samples_jax, copy=True)
        self._np_rng.shuffle(samples_np)
        return torch.from_numpy(samples_np[:num_samples])
