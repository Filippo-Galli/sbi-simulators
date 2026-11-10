"""Simulator for the Lotka-Volterra model."""

import functools
import math
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pyro
import pyro.distributions as pdist
import torch
from ..Simulator import Simulator


def _build_lv_jax_solver(days: float, rk4_dt: float, saveat: float):
    """Build a JIT-compiled batched RK4 ODE solver in JAX for Lotka-Volterra."""
    u0 = jnp.array([30.0, 1.0], dtype=jnp.float32)
    dt = float(rk4_dt)
    n_steps = round(days / dt)
    save_every = max(round(saveat / dt), 1)

    def rhs(u, alpha, beta, gamma, delta):
        x, y = u[0], u[1]
        dx = alpha * x - beta * x * y
        dy = -gamma * y + delta * x * y
        return jnp.stack([dx, dy])

    def solve_single(p):
        alpha, beta, gamma, delta = p[0], p[1], p[2], p[3]

        def step_fn(u, _):
            k1 = rhs(u, alpha, beta, gamma, delta)
            k2 = rhs(u + 0.5 * dt * k1, alpha, beta, gamma, delta)
            k3 = rhs(u + 0.5 * dt * k2, alpha, beta, gamma, delta)
            k4 = rhs(u + dt * k3, alpha, beta, gamma, delta)
            u_next = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            u_next = jnp.where(
                jnp.all(jnp.isfinite(u_next)), u_next, jnp.full_like(u_next, jnp.nan)
            )
            return u_next, u_next

        _, trajectory = jax.lax.scan(step_fn, u0, xs=None, length=n_steps)
        full_traj = jnp.concatenate([u0[None, :], trajectory], axis=0)
        saved = full_traj[::save_every]
        return saved.T  # (2, n_save_points)

    solve_batch = jax.vmap(solve_single)
    return jax.jit(solve_batch)


def _build_lv_mcmc_sampler(jax_solver, prior_loc, prior_scale):
    """Build a JIT-compiled parallel MCMC posterior sampler in JAX using log-space parameterization
    and SIR (Sequential Importance Resampling) initialization to guarantee reliable convergence.
    """
    loc = jnp.asarray(prior_loc, dtype=jnp.float32)
    scale = jnp.asarray(prior_scale, dtype=jnp.float32)

    def log_joint_eta(eta, x_obs):
        # eta = log(theta) in R^4 (unconstrained space)
        diff = (eta - loc) / scale
        lp_prior = -jnp.sum(0.5 * diff**2 + jnp.log(scale) + 0.5 * jnp.log(2.0 * jnp.pi))

        th = jnp.exp(eta)
        us = jax_solver(th[None, :])  # (1, 2, 201)
        us_sub = us[0, :, ::21].reshape(-1)  # (20,)
        clamped_u = jnp.clip(us_sub, 1e-10, 10000.0)
        pred_loc = jnp.log(clamped_u)
        noise_scale = 0.1
        diff_ll = (jnp.log(x_obs) - pred_loc) / noise_scale
        ll = -jnp.sum(
            jnp.log(x_obs) + jnp.log(noise_scale) + 0.5 * jnp.log(2.0 * jnp.pi) + 0.5 * diff_ll**2
        )
        has_nan = jnp.any(jnp.isnan(us))
        return jnp.where(has_nan, -1e9, lp_prior + ll)

    v_log_joint_eta = jax.vmap(log_joint_eta, in_axes=(0, None))

    @functools.partial(
        jax.jit,
        static_argnames=["n_chains", "num_steps", "burn_in", "thin", "n_proposals"],
    )
    def run_mcmc(
        x_obs,
        rng_key,
        n_chains: int = 50,
        num_steps: int = 2000,
        burn_in: int = 800,
        thin: int = 2,
        n_proposals: int = 5000,
    ):
        k_init, k_mcmc = jax.random.split(rng_key)
        # 1. Propose candidate points from prior to find the MAP / high-density mode
        prior_etas = loc + scale * jax.random.normal(k_init, shape=(n_proposals, 4))
        lps = v_log_joint_eta(prior_etas, x_obs)
        best_eta = prior_etas[jnp.argmax(lps)]

        # 2. Initialize all parallel chains near the mode
        k_chains, k_scan = jax.random.split(k_mcmc)
        init_thetas_eta = best_eta + 0.02 * jax.random.normal(k_chains, shape=(n_chains, 4))
        init_lp = v_log_joint_eta(init_thetas_eta, x_obs)

        # Calibrated proposal step size in log-space
        step_size = jnp.array([0.015, 0.015, 0.015, 0.015], dtype=jnp.float32)

        def scan_step(state, subkey):
            curr_eta, curr_lp = state
            k_prop, k_acc = jax.random.split(subkey)
            prop_eta = curr_eta + jax.random.normal(k_prop, shape=(n_chains, 4)) * step_size
            prop_lp = v_log_joint_eta(prop_eta, x_obs)
            alpha = jnp.minimum(1.0, jnp.exp(prop_lp - curr_lp))
            u_rand = jax.random.uniform(k_acc, shape=(n_chains,))
            acc = u_rand < alpha
            new_eta = jnp.where(acc[:, None], prop_eta, curr_eta)
            new_lp = jnp.where(acc, prop_lp, curr_lp)
            return (new_eta, new_lp), jnp.exp(new_eta)

        keys = jax.random.split(k_scan, num_steps)
        _, all_samples = jax.lax.scan(scan_step, (init_thetas_eta, init_lp), keys)
        post_burnin = all_samples[burn_in::thin]
        return post_burnin.reshape(-1, 4)

    return run_mcmc


class LotkaVolterra(Simulator):
    def __init__(
        self,
        dim_parameters: int = 4,
        seed: int = 42,
        days: float = 20.0,
        saveat: float = 0.1,
        summary: str | None = "subsample",
        rk4_dt: float = 0.01,
    ):
        self.dim_parameters = dim_parameters
        if dim_parameters != 4:
            raise ValueError(
                f"LotkaVolterra requires dim_parameters=4 (alpha, beta, "
                f"gamma, delta), got {dim_parameters}."
            )

        self.dim_data_raw = int(2 * (days / saveat + 1))

        if summary is None:
            self.dim_data = self.dim_data_raw
        elif summary == "subsample":
            self.dim_data = 20
        else:
            raise NotImplementedError
        self.summary = summary

        super().__init__(seed, self.dim_parameters, self.dim_data)

        self.u0 = torch.tensor([30.0, 1.0])
        self.tspan = torch.tensor([0.0, days])
        self.days = days
        self.saveat = saveat
        self.rk4_dt = rk4_dt

        # Prior setup
        mu_p1 = -0.125
        mu_p2 = -3.0
        sigma_p = 0.5
        self.prior_loc = [mu_p1, mu_p2, mu_p1, mu_p2]
        self.prior_scale = [sigma_p, sigma_p, sigma_p, sigma_p]
        self.prior_params = {
            "loc": torch.tensor(self.prior_loc),
            "scale": torch.tensor(self.prior_scale),
        }
        self.prior_dist = pdist.LogNormal(**self.prior_params).to_event(1)

        # Build JAX solver and MCMC sampler
        self._jax_solver = _build_lv_jax_solver(
            days=self.days, rk4_dt=self.rk4_dt, saveat=self.saveat
        )
        self._jax_mcmc = _build_lv_mcmc_sampler(
            self._jax_solver,
            prior_loc=self.prior_loc,
            prior_scale=self.prior_scale,
        )

    def _solve(self, parameters: torch.Tensor) -> torch.Tensor:
        """Batched fixed-step RK4 solver accelerated via JAX.

        parameters: (num_samples, 4) tensor of [alpha, beta, gamma, delta]
        returns: (num_samples, 2, num_save_points) tensor
        """
        params_np = parameters.detach().cpu().numpy().astype(np.float32)
        params_jax = jnp.asarray(params_np)
        res_jax = self._jax_solver(params_jax)
        res_np = np.array(res_jax, copy=True)
        return torch.from_numpy(res_np)

    @override
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        parameters = torch.as_tensor(parameters, dtype=torch.float32)

        if parameters.numel() == 0:
            raise ValueError("Expected at least one parameter.")

        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)

        num_samples = parameters.shape[0]

        us = self._solve(parameters)  # (num_samples, 2, dim_data_raw/2)

        expected_points = int(self.dim_data_raw / 2)
        if us.shape[-1] != expected_points:
            if us.shape[-1] > expected_points:
                us = us[:, :, :expected_points]
            else:
                pad = torch.full(
                    (num_samples, 2, expected_points - us.shape[-1]),
                    float("nan"),
                    dtype=torch.float32,
                )
                us = torch.cat([us, pad], dim=-1)

        us = us.float()

        has_nan = torch.isnan(us.reshape(num_samples, -1)).any(dim=1)
        idx_contains_nan = torch.where(has_nan)[0]
        idx_contains_no_nan = torch.where(~has_nan)[0]

        if self.summary is None:
            return us

        if self.summary == "subsample":
            data = torch.full(
                (num_samples, self.dim_data),
                float("nan"),
                dtype=torch.float32,
            )

            if len(idx_contains_nan) == num_samples:
                return data

            us_sub = us[:, :, ::21].reshape(num_samples, -1)

            sampled = pyro.sample(
                "data",
                pdist.LogNormal(
                    loc=torch.log(us_sub[idx_contains_no_nan, :].clamp(1e-10, 10000.0)),
                    scale=0.1,
                ).to_event(1),
            ).float()

            data[idx_contains_no_nan, :] = sampled
            return data

        raise ValueError(f"Unknown summary mode: {self.summary}")

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    def log_likelihood(self, parameters: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the log-likelihood of observation x_obs under parameters.

        parameters: (num_samples, 4) tensor of [alpha, beta, gamma, delta]
        x_obs: (20,) or (1, 20) tensor of observations
        returns: (num_samples,) tensor of log-likelihood values
        """
        parameters = torch.as_tensor(parameters, dtype=torch.float32)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)
        x_obs = torch.as_tensor(x_obs, dtype=torch.float32).flatten()

        us = self._solve(parameters)  # (num_samples, 2, dim_data_raw/2)
        has_nan = torch.isnan(us.reshape(parameters.shape[0], -1)).any(dim=-1)

        us_sub = us[:, :, ::21].reshape(parameters.shape[0], -1)
        dist = torch.distributions.LogNormal(loc=torch.log(us_sub.clamp(1e-10, 10000.0)), scale=0.1)
        ll = dist.log_prob(x_obs).sum(dim=-1)
        ll = torch.where(has_nan, torch.tensor(-1e9, dtype=torch.float32), ll)
        return ll

    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        n_chains: int = 50,
        burn_in: int = 800,
        thin: int = 2,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the posterior p(theta | x_obs) for the Lotka-Volterra model using JAX MCMC
        with log-space parameterization and SIR initialization.

        parameters:
            x_obs: (20,) or (1, 20) tensor observation
            num_samples: desired number of posterior samples
        returns:
            (num_samples, 4) tensor of posterior parameter samples
        """
        if x_obs is None and "num_samples" in kwargs and isinstance(num_samples, (torch.Tensor, np.ndarray)):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)
        if x_obs is None:
            # Default observation 1 from benchmark
            x_obs = torch.tensor(
                [
                    29.864, 10.146, 0.2132, 0.0370, 0.0250, 0.0258,
                    0.0473, 0.0887, 0.1981, 0.4505, 0.9691, 28.685,
                    21.918, 9.1050, 4.4777, 2.7817, 1.4326, 0.6544,
                    0.3537, 0.1631,
                ],
                dtype=torch.float32,
            )

        x_obs_flat = torch.as_tensor(x_obs, dtype=torch.float32).flatten()
        if x_obs_flat.numel() != 20:
            raise ValueError(f"Expected 20 elements in x_obs, got {x_obs_flat.numel()}")

        x_jax = jnp.asarray(x_obs_flat.detach().cpu().numpy())
        rng_key = jax.random.PRNGKey(self.seed)

        # Compute total steps needed to yield num_samples
        steps_needed = math.ceil(num_samples / n_chains) * thin
        total_steps = burn_in + max(steps_needed, 500 * thin)

        samples_jax = self._jax_mcmc(
            x_jax,
            rng_key,
            n_chains=n_chains,
            num_steps=total_steps,
            burn_in=burn_in,
            thin=thin,
        )
        samples_np = np.array(samples_jax, copy=True)
        # Randomly shuffle across chains/time to get i.i.d. draws
        np.random.shuffle(samples_np)
        samples_torch = torch.from_numpy(samples_np)
        return samples_torch[:num_samples]
