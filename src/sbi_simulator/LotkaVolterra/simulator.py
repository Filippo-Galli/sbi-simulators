"""Simulator for the Lotka-Volterra SDE model."""

import math
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pyro.distributions as pdist
import torch

from ..compositional import sample_two_step
from ..mcmc import build_gibbs_pmmh_sampler, build_particle_filter_ll
from ..Simulator import Simulator


def _build_lv_step_fn(
    dt: float = 0.02,
    inner_steps: int = 5,
    sigma1: float = 0.1,
    sigma2: float = 0.1,
):
    """Build single observation step transition function in JAX for Lotka-Volterra."""

    def drift(x, alpha, beta, gamma, delta):
        dx0 = alpha * x[0] - beta * x[0] * x[1]
        dx1 = -gamma * x[1] + delta * x[0] * x[1]
        return jnp.stack([dx0, dx1])

    def integrate_step(x, key, alpha, beta, gamma, delta):
        x = jnp.clip(x, 0.0)
        xdt = drift(x, alpha, beta, gamma, delta)
        dWt = jax.random.normal(key, shape=(2,)) * jnp.sqrt(dt)
        B = jnp.array([sigma1 * x[0] * x[1], sigma2 * x[1] * x[0]])
        x = x + xdt * dt + B * dWt
        return jnp.clip(x, 0.0)

    def step_one(x, key, theta):
        alpha = jax.nn.sigmoid(theta[0]) * 2.0 + 1.0
        beta = jax.nn.sigmoid(theta[1]) * 2.0 + 1.0
        gamma = jax.nn.sigmoid(theta[2]) * 2.0 + 1.0
        delta = jax.nn.sigmoid(theta[3]) * 2.0 + 1.0

        keys = jax.random.split(key, inner_steps)

        def inner_loop(i, curr_x):
            return integrate_step(curr_x, keys[i], alpha, beta, gamma, delta)

        return jax.lax.fori_loop(0, inner_steps, inner_loop, x)

    return step_one


def _build_lv_sde_jax_solver(step_one_fn, num_steps: int = 11):
    """Build a JIT-compiled batched Euler-Maruyama SDE solver in JAX for Lotka-Volterra."""
    default_x0 = jnp.array([1.0, 0.5], dtype=jnp.float32)

    def solve_single(rng_key, theta, x0=None):
        u0 = default_x0 if x0 is None else jnp.clip(x0, 0.0)

        def one_step(x, key):
            x_next = step_one_fn(x, key, theta)
            return x_next, x_next

        keys = jax.random.split(rng_key, num_steps - 1)
        _, xs = jax.lax.scan(one_step, u0, keys)
        full_traj = jnp.concatenate([u0[None, :], xs], axis=0)  # (num_steps, 2)
        return full_traj.T  # (2, num_steps)

    solve_batch_fn = jax.vmap(solve_single, in_axes=(0, 0, 0))

    def solve_batch(rng_keys, thetas, x0s=None):
        if x0s is None:
            x0s = jnp.broadcast_to(default_x0, (thetas.shape[0], 2))
        return solve_batch_fn(rng_keys, thetas, x0s)

    return jax.jit(solve_batch), jax.jit(solve_single)


class LotkaVolterra(Simulator):
    """Lotka-Volterra Stochastic Differential Equation (SDE) simulator.

    Formulation:
        dx0 = (alpha * x0 - beta * x0 * x1) * dt + sigma1 * x0 * x1 * dW0
        dx1 = (-gamma * x1 + delta * x0 * x1) * dt + sigma2 * x1 * x0 * dW1

    Parameters:
        theta: 4D vector in R^4 with standard normal prior N(0, I_4).
        Mapped to positive interaction rates via:
            rate_i = 2.0 * sigmoid(theta_i) + 1.0 in [1.0, 3.0]
    """

    def __init__(
        self,
        dim_parameters: int = 4,
        seed: int = 42,
        dt: float = 0.02,
        inner_steps: int = 5,
        sigma1: float = 0.1,
        sigma2: float = 0.1,
        num_steps: int = 11,
        obs_sigma: float = 0.05,
        summary: str | None = None,
    ):
        self.dim_parameters = dim_parameters
        if dim_parameters != 4:
            raise ValueError(
                f"LotkaVolterra requires dim_parameters=4 (alpha, beta, "
                f"gamma, delta), got {dim_parameters}."
            )

        self.dt = dt
        self.inner_steps = inner_steps
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.num_steps = num_steps
        self.obs_sigma = obs_sigma
        self.dim_data_raw = int(2 * num_steps)

        if summary is None or summary == "subsample":
            self.dim_data = self.dim_data_raw
        elif summary in ("stats", "summary_stats"):
            self.dim_data = 9
        elif summary == "two_step":
            self.dim_data = 4
        else:
            raise NotImplementedError(f"Unknown summary mode: {summary}")
        self.summary = summary

        super().__init__(seed, self.dim_parameters, self.dim_data)

        self.u0 = torch.tensor([1.0, 0.5], dtype=torch.float32)

        # Prior setup: Standard Normal N(0, I_4) matching markovsbi
        self.prior_loc = [0.0, 0.0, 0.0, 0.0]
        self.prior_scale = [1.0, 1.0, 1.0, 1.0]
        self.prior_dist = pdist.Normal(
            loc=torch.zeros(4, dtype=torch.float32),
            scale=torch.ones(4, dtype=torch.float32),
        ).to_event(1)

        # Build JAX SDE solver and Particle Filter MCMC sampler
        self._step_fn = _build_lv_step_fn(
            dt=self.dt,
            inner_steps=self.inner_steps,
            sigma1=self.sigma1,
            sigma2=self.sigma2,
        )
        self._jax_solver_batch, self._jax_solver_single = _build_lv_sde_jax_solver(
            self._step_fn,
            num_steps=self.num_steps,
        )
        self._pf_ll_fn = build_particle_filter_ll(
            self._step_fn,
            obs_dim=2,
            obs_sigma=self.obs_sigma,
        )
        self._jax_mcmc = build_gibbs_pmmh_sampler(
            self._pf_ll_fn,
            dim_parameters=self.dim_parameters,
        )

    def _solve(
        self,
        parameters: torch.Tensor,
        u0: torch.Tensor | None = None,
        rng_key: jax.Array | None = None,
    ) -> torch.Tensor:
        """Batched SDE Euler-Maruyama simulation accelerated via JAX.

        parameters: (num_samples, 4) tensor of raw thetas
        u0: optional (num_samples, 2) tensor of initial states
        returns: (num_samples, 2, num_steps) tensor
        """
        params_np = parameters.detach().cpu().numpy().astype(np.float32)
        params_jax = jnp.asarray(params_np)
        batch_size = params_jax.shape[0]

        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        keys = jax.random.split(rng_key, batch_size)

        if u0 is not None:
            u0_np = u0.detach().cpu().numpy().astype(np.float32)
            u0_jax = jnp.asarray(u0_np)
            res_jax = self._jax_solver_batch(keys, params_jax, u0_jax)
        else:
            res_jax = self._jax_solver_batch(keys, params_jax)

        res_np = np.array(res_jax, copy=True)
        return torch.from_numpy(res_np)

    def get_prior_x0(self) -> pdist.Distribution:
        """Prior over initial condition x0 = [prey, predator]."""
        return pdist.Uniform(
            low=torch.tensor([0.0, 0.0], dtype=torch.float32),
            high=torch.tensor([10.0, 10.0], dtype=torch.float32),
        ).to_event(1)

    @staticmethod
    def compute_summary_stats(us: torch.Tensor) -> torch.Tensor:
        """Compute 9 summary statistics from (num_samples, 2, T) trajectory tensor."""
        is_2d_traj = us.ndim == 2
        if is_2d_traj:
            us = us.unsqueeze(0)

        eps = 1e-6
        x = us[:, 0, :]
        y = us[:, 1, :]

        mean_x = x.mean(dim=-1, keepdim=True)
        mean_y = y.mean(dim=-1, keepdim=True)

        dx = x - mean_x
        dy = y - mean_y

        var_x = (dx**2).mean(dim=-1, keepdim=True)
        var_y = (dy**2).mean(dim=-1, keepdim=True)

        log_var_x = torch.log(var_x.clamp_min(eps))
        log_var_y = torch.log(var_y.clamp_min(eps))

        T = us.shape[-1]
        denom_x = var_x.clamp_min(eps)
        denom_y = var_y.clamp_min(eps)

        if T >= 2:
            ac1_x = (dx[:, 1:] * dx[:, :-1]).mean(dim=-1, keepdim=True) / denom_x
            ac1_y = (dy[:, 1:] * dy[:, :-1]).mean(dim=-1, keepdim=True) / denom_y
        else:
            ac1_x = torch.zeros_like(mean_x)
            ac1_y = torch.zeros_like(mean_y)

        if T >= 3:
            ac2_x = (dx[:, 2:] * dx[:, :-2]).mean(dim=-1, keepdim=True) / denom_x
            ac2_y = (dy[:, 2:] * dy[:, :-2]).mean(dim=-1, keepdim=True) / denom_y
        else:
            ac2_x = torch.zeros_like(mean_x)
            ac2_y = torch.zeros_like(mean_y)

        denom_xy = torch.sqrt(denom_x * denom_y)
        cc_xy = (dx * dy).mean(dim=-1, keepdim=True) / denom_xy

        ac1_x = torch.nan_to_num(ac1_x.clamp(-1.0, 1.0), nan=0.0)
        ac2_x = torch.nan_to_num(ac2_x.clamp(-1.0, 1.0), nan=0.0)
        ac1_y = torch.nan_to_num(ac1_y.clamp(-1.0, 1.0), nan=0.0)
        ac2_y = torch.nan_to_num(ac2_y.clamp(-1.0, 1.0), nan=0.0)
        cc_xy = torch.nan_to_num(cc_xy.clamp(-1.0, 1.0), nan=0.0)

        stats = torch.cat(
            [
                mean_x,
                mean_y,
                log_var_x,
                log_var_y,
                ac1_x,
                ac2_x,
                ac1_y,
                ac2_y,
                cc_xy,
            ],
            dim=-1,
        )
        return stats.squeeze(0) if is_2d_traj else stats

    def stepper(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        dt: float | None = None,
        sigma: float | None = None,
    ) -> torch.Tensor:
        """Simulate a single time-series step transition for Lotka-Volterra SDE.

        Parameters
        ----------
        x : torch.Tensor
            Current state [x0, x1], shape (2,) or (batch_size, 2).
        parameters : torch.Tensor
            Lotka-Volterra parameters (raw theta in R^4 or rates),
            shape (4,) or (batch_size, 4).
        dt : float, optional
            Step time interval (default: self.dt * self.inner_steps = 0.1).
        sigma : float, optional
            Diffusion level (default: self.sigma1 = 0.1).

        Returns
        -------
        torch.Tensor
            Next state [x0, x1] of shape (2,) or (batch_size, 2).
        """
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

        sig1 = float(self.sigma1 if sigma is None else sigma)
        sig2 = float(self.sigma2 if sigma is None else sigma)

        # Parameter mapping
        alpha = torch.sigmoid(params[:, 0]) * 2.0 + 1.0
        beta = torch.sigmoid(params[:, 1]) * 2.0 + 1.0
        gamma = torch.sigmoid(params[:, 2]) * 2.0 + 1.0
        delta = torch.sigmoid(params[:, 3]) * 2.0 + 1.0

        step_dt = float(self.dt * self.inner_steps if dt is None else dt)
        sub_dt = float(self.dt)
        n_substeps = max(round(step_dt / sub_dt), 1)
        actual_h = step_dt / n_substeps

        curr_u = torch.clamp(x_t.clone(), min=0.0)

        for _ in range(n_substeps):
            p0 = curr_u[:, 0]
            p1 = curr_u[:, 1]
            dx0 = alpha * p0 - beta * p0 * p1
            dx1 = -gamma * p1 + delta * p0 * p1
            drift_step = torch.stack([dx0, dx1], dim=-1)

            dW = torch.randn_like(curr_u) * math.sqrt(actual_h)
            diff_0 = sig1 * p0 * p1 * dW[:, 0]
            diff_1 = sig2 * p1 * p0 * dW[:, 1]
            diff_step = torch.stack([diff_0, diff_1], dim=-1)

            curr_u = torch.clamp(curr_u + drift_step * actual_h + diff_step, min=0.0)

        return curr_u.squeeze(0) if is_1d else curr_u

    def simulate_two_step(
        self,
        parameters: torch.Tensor,
        x0: torch.Tensor | None = None,
        t: int | torch.Tensor | None = None,
        informed: bool = True,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Generate two-step transition pairs (x_t, x_{t+1}) like in markovsbi."""
        return sample_two_step(
            self,
            parameters=parameters,
            x0=x0,
            t=t,
            informed=informed,
            return_dict=return_dict,
        )

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

        us = self._solve(parameters)  # (num_samples, 2, num_steps)
        us = us.float()

        has_nan = torch.isnan(us.reshape(num_samples, -1)).any(dim=1)
        idx_contains_nan = torch.where(has_nan)[0]
        idx_contains_no_nan = torch.where(~has_nan)[0]

        if self.summary is None:
            # Flatten to (num_samples, dim_data)
            traj_flat = us.reshape(num_samples, -1)
            return traj_flat.squeeze(0) if is_1d else traj_flat

        if self.summary == "subsample":
            data = torch.full(
                (num_samples, self.dim_data),
                float("nan"),
                dtype=torch.float32,
            )

            if len(idx_contains_nan) == num_samples:
                return data.squeeze(0) if is_1d else data

            # Additive Gaussian observation noise
            raw_data = us[idx_contains_no_nan, :, :].reshape(
                len(idx_contains_no_nan), -1
            )
            noise = torch.randn_like(raw_data) * self.obs_sigma
            sampled = raw_data + noise

            data[idx_contains_no_nan, :] = sampled
            return data.squeeze(0) if is_1d else data

        if self.summary in ("stats", "summary_stats"):
            data = torch.full(
                (num_samples, self.dim_data),
                float("nan"),
                dtype=torch.float32,
            )

            if len(idx_contains_nan) == num_samples:
                return data.squeeze(0) if is_1d else data

            stats = self.compute_summary_stats(us[idx_contains_no_nan, :, :])
            data[idx_contains_no_nan, :] = stats
            return data.squeeze(0) if is_1d else data

        raise ValueError(f"Unknown summary mode: {self.summary}")

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def log_likelihood(
        self, parameters: torch.Tensor, x_obs: torch.Tensor, n_particles: int = 150
    ) -> torch.Tensor:
        """Evaluate marginal log-likelihood using JAX Particle Filter."""
        parameters = torch.as_tensor(parameters, dtype=torch.float32)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)

        num_samples = parameters.shape[0]
        x_obs_flat = torch.as_tensor(x_obs, dtype=torch.float32).flatten()
        x_obs_2d = x_obs_flat.reshape(-1, 2)

        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        params_jax = jnp.asarray(parameters.detach().cpu().numpy())

        key = jax.random.PRNGKey(self.seed)
        keys = jax.random.split(key, num_samples)

        v_ll = jax.vmap(
            lambda th, k: self._pf_ll_fn(th, x_jax, k, n_particles=n_particles)
        )
        lls_jax = v_ll(params_jax, keys)
        return torch.from_numpy(np.array(lls_jax, copy=True))

    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        n_chains: int = 20,
        burn_in: int = 200,
        inner_thin: int = 4,
        n_particles: int = 150,
        step_size: float = 0.3,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from posterior p(theta | x_obs) using JAX Gibbs PMMH."""
        if (
            x_obs is None
            and "num_samples" in kwargs
            and isinstance(num_samples, (torch.Tensor, np.ndarray))
        ):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)

        if x_obs is None:
            # Default observation generated from theta = 0 with seed 42
            x_obs = torch.tensor(
                [
                    [1.0000, 0.5000],
                    [1.0762, 0.5163],
                    [1.1869, 0.5604],
                    [1.2694, 0.5928],
                    [1.4117, 0.6073],
                    [1.5249, 0.6987],
                    [1.6904, 0.7705],
                    [1.7563, 0.8746],
                    [1.7599, 1.0564],
                    [1.7599, 1.3159],
                    [1.7551, 1.5372],
                ],
                dtype=torch.float32,
            )

        x_obs_tensor = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_tensor.ndim == 1:
            x_obs_2d = x_obs_tensor.reshape(-1, 2)
        else:
            x_obs_2d = x_obs_tensor

        x_jax = jnp.asarray(x_obs_2d.detach().cpu().numpy())
        rng_key = jax.random.PRNGKey(self.seed)

        num_samples_per_chain = math.ceil(num_samples / n_chains)

        samples_jax = self._jax_mcmc(
            x_jax,
            rng_key,
            n_chains=n_chains,
            num_samples_per_chain=num_samples_per_chain,
            burn_in=burn_in,
            inner_thin=inner_thin,
            n_particles=n_particles,
            step_size=step_size,
        )
        samples_np = np.array(samples_jax, copy=True)
        np.random.shuffle(samples_np)
        samples_torch = torch.from_numpy(samples_np)
        return samples_torch[:num_samples]
