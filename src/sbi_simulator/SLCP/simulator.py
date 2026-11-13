"""Simulator for the Simple Likelihood Complex Posterior (SLCP) benchmark model."""

import functools
import math
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pdist
import torch

from ..compositional import sample_two_step
from ..Simulator import Simulator


def _build_slcp_jax_sampler():
    """Build JIT-compiled log-likelihood and MCMC sampler in JAX for SLCP."""

    def slcp_log_likelihood(theta, x_obs):
        # theta: (5,), x_obs: (4, 2)
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
        ll_per_obs = -0.5 * (quad + 2.0 * log_2pi + log_det)

        return jnp.where(in_bounds, jnp.sum(ll_per_obs), -1e9)

    v_log_lik = jax.vmap(slcp_log_likelihood, in_axes=(0, None))

    @functools.partial(
        jax.jit,
        static_argnames=[
            "n_chains",
            "num_samples_per_chain",
            "burn_in",
            "thin",
            "n_sir",
        ],
    )
    def run_mcmc(
        x_obs,
        rng_key,
        n_chains: int = 20,
        num_samples_per_chain: int = 500,
        burn_in: int = 500,
        thin: int = 4,
        step_size: float = 0.08,
        n_sir: int = 5000,
    ):
        k_init, k_mcmc = jax.random.split(rng_key)

        def init_chain(k):
            k1, k2 = jax.random.split(k)
            candidates = jax.random.uniform(
                k1, shape=(n_sir, 5), minval=-3.0, maxval=3.0
            )
            lps = v_log_lik(candidates, x_obs)
            max_lp = jnp.max(lps)
            weights = jnp.exp(lps - max_lp)
            weights = weights / jnp.sum(weights)
            idx = jax.random.choice(k2, n_sir, p=weights)
            chosen = candidates[idx]
            return chosen, slcp_log_likelihood(chosen, x_obs)

        init_keys = jax.random.split(k_init, n_chains)
        curr_thetas, curr_lps = jax.vmap(init_chain)(init_keys)

        def step_fn(state, key):
            thetas, lps = state
            k_prop, k_acc = jax.random.split(key)
            props = thetas + jax.random.normal(k_prop, shape=thetas.shape) * step_size
            prop_lps = v_log_lik(props, x_obs)
            alpha = jnp.minimum(1.0, jnp.exp(prop_lps - lps))
            acc = jax.random.uniform(k_acc, shape=(n_chains,)) < alpha
            new_thetas = jnp.where(acc[:, None], props, thetas)
            new_lps = jnp.where(acc, prop_lps, lps)
            return (new_thetas, new_lps), new_thetas

        total_steps = burn_in + num_samples_per_chain * thin
        keys = jax.random.split(k_mcmc, total_steps)
        _, all_samples = jax.lax.scan(step_fn, (curr_thetas, curr_lps), keys)
        samples = all_samples[burn_in::thin]  # (num_samples_per_chain, n_chains, 5)
        return samples.transpose(1, 0, 2).reshape(-1, 5)

    return jax.jit(slcp_log_likelihood), run_mcmc


class SLCP(Simulator):
    r"""Simple Likelihood Complex Posterior (SLCP) benchmark simulator.

    Reference:
        Papamakarios et al. (2019), "Sequential Neural Likelihood".
        Lueckmann et al. (2021), "Benchmarking Simulation-Based Inference".

    Parameters:
        \theta \in \mathbb{R}^5 \sim \text{Uniform}(-3, 3)^5

    Likelihood:
        Generates 4 independent 2D observations y_1, ..., y_4 \sim \mathcal{N}(m(\theta), \Sigma(\theta)):
            m(\theta) = [\theta_0, \theta_1]^T
            s_1 = \theta_2^2, \quad s_2 = \theta_3^2, \quad \rho = \tanh(\theta_4)
            \Sigma(\theta) = [[s_1^2 + \epsilon, \rho s_1 s_2],
                              [\rho s_1 s_2,     s_2^2 + \epsilon]]

    The squared parameterization s_1 = \theta_2^2, s_2 = \theta_3^2 creates 4 symmetric modes
    in (\theta_2, \theta_3), producing a challenging multi-modal posterior.
    """

    def __init__(
        self,
        dim_parameters: int = 5,
        seed: int = 42,
        num_data: int = 4,
        summary: str | None = None,
    ):
        self.dim_parameters = dim_parameters
        if dim_parameters != 5:
            raise ValueError(
                f"SLCP requires dim_parameters=5 (theta_0 to theta_4), got {dim_parameters}."
            )

        self.num_data = num_data
        self.dim_data_raw = int(2 * num_data)  # 8

        if summary is None or summary == "subsample":
            self.dim_data = self.dim_data_raw
        elif summary in ("stats", "summary_stats"):
            self.dim_data = 5  # mean_x, mean_y, logvar_x, logvar_y, corr_xy
        elif summary == "two_step":
            self.dim_data = 4  # [x_t, x_{t+1}] in R^4
        else:
            raise NotImplementedError(f"Unknown summary mode: {summary}")
        self.summary = summary

        super().__init__(seed, self.dim_parameters, self.dim_data)

        # Prior distribution setup: Uniform(-3, 3)^5
        self.prior_low = -3.0
        self.prior_high = 3.0
        self.prior_dist = pdist.Uniform(
            low=torch.full((5,), self.prior_low, dtype=torch.float32),
            high=torch.full((5,), self.prior_high, dtype=torch.float32),
        ).to_event(1)

        self.u0 = torch.zeros(2, dtype=torch.float32)

        # Build JAX log-likelihood and fast parallel MCMC sampler
        self._jax_ll_fn, self._jax_mcmc = _build_slcp_jax_sampler()

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial point x0 in R^2."""
        return pdist.Uniform(
            low=torch.tensor([-15.0, -15.0], dtype=torch.float32),
            high=torch.tensor([15.0, 15.0], dtype=torch.float32),
        ).to_event(1)

    def stepper(self, x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """Generate one transition step from parameters."""
        params = torch.as_tensor(parameters, dtype=torch.float32)
        is_1d = params.ndim == 1
        if is_1d:
            params = params.unsqueeze(0)
        batch_size = params.shape[0]

        m = params[:, :2]
        s1 = params[:, 2] ** 2
        s2 = params[:, 3] ** 2
        rho = torch.tanh(params[:, 4])

        eps = 1e-6
        var1 = s1**2 + eps
        var2 = s2**2 + eps
        cov = rho * s1 * s2

        S = torch.empty((batch_size, 2, 2), dtype=torch.float32)
        S[:, 0, 0] = var1
        S[:, 0, 1] = cov
        S[:, 1, 0] = cov
        S[:, 1, 1] = var2

        dist = pdist.MultivariateNormal(m, S)
        next_x = dist.sample()
        return next_x.squeeze(0) if is_1d else next_x

    @staticmethod
    def compute_summary_stats(xs: torch.Tensor) -> torch.Tensor:
        """Compute 5 summary statistics from (num_samples, 4, 2) observation tensor.

        Statistics:
            - Mean of dimension 0
            - Mean of dimension 1
            - Log-variance of dimension 0
            - Log-variance of dimension 1
            - Correlation coefficient between dimension 0 and 1
        """
        is_2d = xs.ndim == 2  # (4, 2)
        if is_2d:
            xs = xs.unsqueeze(0)  # (1, 4, 2)

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

    def simulate_two_step(
        self,
        parameters: torch.Tensor,
        x0: torch.Tensor | None = None,
        t: int | torch.Tensor | None = None,
        informed: bool = True,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Generate two-step transition pairs (x_t, x_{t+1}) for compositional SBI."""
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
        """Simulate sequence of 4 2D observations given parameters.

        parameters: (batch_size, 5) tensor
        returns: (batch_size, 4, 2) tensor of 4 points
        """
        params = torch.as_tensor(parameters, dtype=torch.float32)
        is_1d = params.ndim == 1
        if is_1d:
            params = params.unsqueeze(0)
        batch_size = params.shape[0]

        m = params[:, :2]
        s1 = params[:, 2] ** 2
        s2 = params[:, 3] ** 2
        rho = torch.tanh(params[:, 4])

        eps = 1e-6
        var1 = s1**2 + eps
        var2 = s2**2 + eps
        cov = rho * s1 * s2

        S = torch.empty((batch_size, 2, 2), dtype=torch.float32)
        S[:, 0, 0] = var1
        S[:, 0, 1] = cov
        S[:, 1, 0] = cov
        S[:, 1, 1] = var2

        dist = pdist.MultivariateNormal(m.unsqueeze(1), S.unsqueeze(1)).expand(
            (batch_size, self.num_data)
        )
        data = dist.sample()  # (batch_size, 4, 2)
        return data.squeeze(0) if is_1d else data

    @override
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        parameters = torch.as_tensor(parameters, dtype=torch.float32)
        if parameters.numel() == 0:
            raise ValueError("Expected at least one parameter.")

        is_1d = parameters.ndim == 1
        if is_1d:
            parameters = parameters.unsqueeze(0)
        num_samples = parameters.shape[0]

        if self.summary == "two_step":
            pairs = self.simulate_two_step(parameters, return_dict=False)
            pairs_flat = pairs.reshape(num_samples, 4)
            return pairs_flat.squeeze(0) if is_1d else pairs_flat

        xs = self._solve(parameters)  # (num_samples, 4, 2)

        if self.summary is None or self.summary == "subsample":
            xs_flat = xs.reshape(num_samples, self.dim_data_raw)
            return xs_flat.squeeze(0) if is_1d else xs_flat

        if self.summary in ("stats", "summary_stats"):
            stats = self.compute_summary_stats(xs)
            return stats.squeeze(0) if is_1d else stats

        raise ValueError(f"Unknown summary mode: {self.summary}")

    def log_likelihood(
        self, parameters: torch.Tensor, x_obs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate exact analytical log-likelihood of observation x_obs under parameters."""
        parameters = torch.as_tensor(parameters, dtype=torch.float32)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)

        x_obs_flat = torch.as_tensor(x_obs, dtype=torch.float32).flatten()
        x_obs_2d = x_obs_flat.reshape(self.num_data, 2)

        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        params_jax = jnp.asarray(parameters.detach().cpu().numpy())

        v_ll = jax.vmap(lambda th: self._jax_ll_fn(th, x_jax))
        lls_jax = v_ll(params_jax)
        return torch.from_numpy(np.array(lls_jax, copy=True))

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
        """Sample from the true posterior p(theta | x_obs) using JAX MCMC."""
        if (
            x_obs is None
            and "num_samples" in kwargs
            and isinstance(num_samples, (torch.Tensor, np.ndarray))
        ):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)

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

        x_obs_tensor = torch.as_tensor(x_obs, dtype=torch.float32)
        x_obs_2d = x_obs_tensor.flatten().reshape(self.num_data, 2)
        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())

        rng_key = jax.random.PRNGKey(self.seed)
        num_samples_per_chain = math.ceil(num_samples / n_chains)

        samples_jax = self._jax_mcmc(
            x_jax,
            rng_key,
            n_chains=n_chains,
            num_samples_per_chain=num_samples_per_chain,
            burn_in=burn_in,
            thin=thin,
            step_size=step_size,
            n_sir=n_sir,
        )

        samples_np = np.array(samples_jax, copy=True)
        np.random.shuffle(samples_np)
        samples_torch = torch.from_numpy(samples_np)
        return samples_torch[:num_samples]
