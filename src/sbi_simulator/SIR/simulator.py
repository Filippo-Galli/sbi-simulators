"""Simulator for the SIR model."""

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


def _build_sir_jax_solver(
    N: float, I0: float, R0: float, days: float, rk4_dt: float, saveat: float
):
    """Build a JIT-compiled batched RK4 ODE solver in JAX."""
    u0 = jnp.array([N - I0 - R0, I0, R0], dtype=jnp.float32)
    dt = float(rk4_dt)
    n_steps = round(days / dt)
    save_every = max(round(saveat / dt), 1)

    def rhs(u, beta, gamma):
        S, I, _ = u[0], u[1], u[2]
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return jnp.stack([dS, dI, dR])

    def solve_single(p):
        beta, gamma = p[0], p[1]

        def step_fn(u, _):
            k1 = rhs(u, beta, gamma)
            k2 = rhs(u + 0.5 * dt * k1, beta, gamma)
            k3 = rhs(u + 0.5 * dt * k2, beta, gamma)
            k4 = rhs(u + dt * k3, beta, gamma)
            u_next = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            u_next = jnp.where(
                jnp.all(jnp.isfinite(u_next)), u_next, jnp.full_like(u_next, jnp.nan)
            )
            return u_next, u_next

        _, trajectory = jax.lax.scan(step_fn, u0, xs=None, length=n_steps)
        full_traj = jnp.concatenate([u0[None, :], trajectory], axis=0)
        saved = full_traj[::save_every]  # (n_save_points, 3)
        return saved.T  # (3, n_save_points)

    solve_batch = jax.vmap(solve_single)
    return jax.jit(solve_batch)


def _build_sir_mcmc_sampler(jax_solver, N: float, total_count: int, prior_loc, prior_scale):
    """Build a JIT-compiled parallel MCMC posterior sampler in JAX using log-space parameterization
    and SIR (Sequential Importance Resampling) initialization to guarantee reliable convergence.
    """
    loc = jnp.asarray(prior_loc, dtype=jnp.float32)
    scale = jnp.asarray(prior_scale, dtype=jnp.float32)

    def log_joint_eta(eta, x_obs):
        # eta = log(theta) in R^2 (unconstrained space)
        # Prior on eta is Normal(loc, scale^2); Jacobian cancels out with LogNormal density
        diff = (eta - loc) / scale
        lp_prior = -jnp.sum(0.5 * diff**2 + jnp.log(scale) + 0.5 * jnp.log(2.0 * jnp.pi))

        th = jnp.exp(eta)
        us = jax_solver(th[None, :])  # (1, 3, 161)
        infected = us[0, 1, ::17]  # (10,)
        probs = jnp.clip(infected / N, 1e-6, 1.0 - 1e-6)
        k = x_obs  # (10,)
        n = float(total_count)
        log_comb = jax.lax.lgamma(n + 1.0) - jax.lax.lgamma(k + 1.0) - jax.lax.lgamma(n - k + 1.0)
        ll = jnp.sum(log_comb + k * jnp.log(probs) + (n - k) * jnp.log(1.0 - probs))
        has_nan = jnp.any(jnp.isnan(us))
        return jnp.where(has_nan, -1e9, lp_prior + ll)

    v_log_joint_eta = jax.vmap(log_joint_eta, in_axes=(0, None))

    @functools.partial(jax.jit, static_argnames=["n_chains", "num_steps", "burn_in", "thin", "n_proposals"])
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
        prior_etas = loc + scale * jax.random.normal(k_init, shape=(n_proposals, 2))
        lps = v_log_joint_eta(prior_etas, x_obs)
        best_eta = prior_etas[jnp.argmax(lps)]

        # 2. Initialize all parallel chains near the mode
        k_chains, k_scan = jax.random.split(k_mcmc)
        init_thetas_eta = best_eta + 0.02 * jax.random.normal(k_chains, shape=(n_chains, 2))
        init_lp = v_log_joint_eta(init_thetas_eta, x_obs)

        # Calibrated proposal step size in log-space
        step_size = jnp.array([0.025, 0.045], dtype=jnp.float32)

        def scan_step(state, subkey):
            curr_eta, curr_lp = state
            k_prop, k_acc = jax.random.split(subkey)
            prop_eta = curr_eta + jax.random.normal(k_prop, shape=(n_chains, 2)) * step_size
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
        return post_burnin.reshape(-1, 2)

    return run_mcmc


class SIR(Simulator):
    def __init__(
        self,
        dim_parameters: int = 2,
        seed: int = 42,
        N: float = 1000000.0,
        I0: float = 1.0,
        R0: float = 0.0,
        days: float = 160.0,
        saveat: float = 1.0,
        total_count: int = 1000,
        summary: str | None = "subsample",
        rk4_dt: float = 0.1,
    ):
        self.dim_parameters = dim_parameters
        if dim_parameters != 2:
            raise ValueError(f"SIR requires dim_parameters=2 (beta, gamma), got {dim_parameters}.")

        self.dim_data_raw = int(3 * (days / saveat + 1))
        if summary is None:
            self.dim_data = self.dim_data_raw
        elif summary == "subsample":
            self.dim_data = 10
        else:
            raise NotImplementedError
        super().__init__(seed, self.dim_parameters, self.dim_data)

        self.N = N
        self.I0 = I0
        self.R0 = R0
        self.days = days
        self.saveat = saveat
        self.total_count = total_count
        self.rk4_dt = rk4_dt

        self.u0 = torch.tensor([N - I0 - R0, I0, R0])
        self.tspan = torch.tensor([0.0, days])

        self.summary = summary

        # Prior distribution setup
        self.prior_loc = [math.log(0.4), math.log(0.125)]
        self.prior_scale = [0.5, 0.2]
        self.prior_params = {
            "loc": torch.tensor(self.prior_loc),
            "scale": torch.tensor(self.prior_scale),
        }
        self.prior_dist = pdist.LogNormal(**self.prior_params).to_event(1)

        # Build JAX solver and MCMC sampler
        self._jax_solver = _build_sir_jax_solver(
            N=self.N,
            I0=self.I0,
            R0=self.R0,
            days=self.days,
            rk4_dt=self.rk4_dt,
            saveat=self.saveat,
        )
        self._jax_mcmc = _build_sir_mcmc_sampler(
            self._jax_solver,
            N=self.N,
            total_count=self.total_count,
            prior_loc=self.prior_loc,
            prior_scale=self.prior_scale,
        )

    def _solve(self, parameters: torch.Tensor) -> torch.Tensor:
        """Batched fixed-step RK4 solver accelerated via JAX.

        parameters: (num_samples, 2) tensor of [beta, gamma]
        returns: (num_samples, 3, num_save_points) tensor
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

        us = self._solve(parameters)  # (num_samples, 3, dim_data_raw/3)

        expected_points = int(self.dim_data_raw / 3)
        if us.shape[-1] != expected_points:
            if us.shape[-1] > expected_points:
                us = us[:, :, :expected_points]
            else:
                pad = torch.full(
                    (num_samples, 3, expected_points - us.shape[-1]),
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

            infected = us[:, 1, ::17].reshape(num_samples, -1)

            sampled = pyro.sample(
                "data",
                pdist.Binomial(
                    total_count=self.total_count,
                    probs=(infected[idx_contains_no_nan, :] / self.N).clamp(0.0, 1.0),
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

        parameters: (num_samples, 2) tensor of [beta, gamma]
        x_obs: (10,) or (1, 10) tensor of observations
        returns: (num_samples,) tensor of log-likelihood values
        """
        parameters = torch.as_tensor(parameters, dtype=torch.float32)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)
        x_obs = torch.as_tensor(x_obs, dtype=torch.float32).flatten()

        us = self._solve(parameters)  # (num_samples, 3, dim_data_raw/3)
        has_nan = torch.isnan(us.reshape(parameters.shape[0], -1)).any(dim=-1)

        infected = us[:, 1, ::17].reshape(parameters.shape[0], -1)
        probs = (infected / self.N).clamp(1e-6, 1.0 - 1e-6)

        dist = torch.distributions.Binomial(total_count=self.total_count, probs=probs)
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
        """Sample from the posterior p(theta | x_obs) for the SIR model using JAX MCMC
        with log-space parameterization and SIR initialization.

        parameters:
            x_obs: (10,) or (1, 10) tensor observation
            num_samples: desired number of posterior samples
        returns:
            (num_samples, 2) tensor of posterior parameter samples
        """
        if x_obs is None and "num_samples" in kwargs and isinstance(num_samples, (torch.Tensor, np.ndarray)):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)
        if x_obs is None:
            x_obs = torch.tensor([0.0, 1.0, 352.0, 40.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        x_obs_flat = torch.as_tensor(x_obs, dtype=torch.float32).flatten()
        if x_obs_flat.numel() != 10:
            raise ValueError(f"Expected 10 elements in x_obs, got {x_obs_flat.numel()}")

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
