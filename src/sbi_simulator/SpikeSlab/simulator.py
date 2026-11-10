"""Simulator for the Spike-and-Slab mixture model."""

from typing import override

import numpy as np
import pyro.distributions as pdist
import torch
from ..Simulator import Simulator


class SpikeSlab(Simulator):
    def __init__(
        self,
        seed: int = 42,
        dim_parameters: int = 2,
        dim_data: int | None = None,
    ):
        if dim_data is None:
            dim_data = dim_parameters
        super().__init__(seed=seed, dim_parameters=dim_parameters, dim_data=dim_data)

    @override
    def get_prior(self) -> pdist.Distribution:
        return pdist.Uniform(
            -10.0 * torch.ones(self.dim_parameters),
            10.0 * torch.ones(self.dim_parameters),
        ).to_event(1)

    @override
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        params = torch.as_tensor(parameters, dtype=torch.float32)
        if params.numel() == 0:
            raise ValueError("Expected at least one parameter.")

        is_1d = params.ndim == 1
        if is_1d:
            params = params.unsqueeze(0)

        batch_size = params.shape[0]
        choose_broad = torch.rand((batch_size, 1), device=params.device) < 0.5
        std = torch.where(choose_broad, 1.0, 0.1)

        out = params + std * torch.randn_like(params)
        return out.squeeze(0) if is_1d else out

    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the exact posterior for the SpikeSlab (Gaussian-mixture) simulator
        with a uniform prior on [-10, 10]^D.
        For a single observation x_obs:
            p(x|theta) = 0.5 N(x; theta, I) + 0.5 N(x; theta, 0.01 I),
            p(theta|x) ∝ [0.5 N(theta; x, I) + 0.5 N(theta; x, 0.01 I)] * I(theta in [-10, 10]^D).
        """
        if x_obs is None and "num_samples" in kwargs and isinstance(num_samples, (torch.Tensor, np.ndarray)):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)
        if x_obs is None:
            x_obs = torch.zeros(self.dim_data, dtype=torch.float32)

        x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_t.ndim == 1:
            x_obs_t = x_obs_t.unsqueeze(0)

        n_obs = x_obs_t.shape[0]
        d = x_obs_t.shape[-1]
        lo, hi = -10.0, 10.0

        if n_obs == 1:
            x_single = x_obs_t[0]
            out = []
            collected = 0
            while collected < num_samples:
                needed = num_samples - collected
                batch = max(2 * needed, 1000)
                choose_broad = torch.rand((batch, 1)) < 0.5
                stds = torch.where(choose_broad, 1.0, 0.1)
                candidates = x_single.unsqueeze(0) + stds * torch.randn(batch, d)
                keep = torch.all((candidates >= lo) & (candidates <= hi), dim=-1)
                valid = candidates[keep]
                if valid.shape[0] > 0:
                    out.append(valid)
                    collected += valid.shape[0]
            samples = torch.cat(out, dim=0)[:num_samples]
            return samples

        # For n_obs > 1, evaluate exact product of mixtures log-likelihood and sample via MCMC
        # Likelihood of each observation under theta:
        def log_joint(th):
            # th shape: (K, D)
            # diff shape: (K, n_obs, D)
            diff = x_obs_t.unsqueeze(0) - th.unsqueeze(1)
            dist_sq = diff.square().sum(dim=-1)  # (K, n_obs)
            ll1 = -0.5 * (d * np.log(2.0 * np.pi) + dist_sq)
            ll2 = -0.5 * (d * np.log(2.0 * np.pi * 0.01) + dist_sq / 0.01)
            ll_comp = torch.stack([ll1, ll2], dim=-1)  # (K, n_obs, 2)
            ll_per_obs = torch.logsumexp(ll_comp + np.log(0.5), dim=-1)  # (K, n_obs)
            total_ll = ll_per_obs.sum(dim=-1)  # (K,)
            in_prior = torch.all((th >= lo) & (th <= hi), dim=-1)
            return torch.where(in_prior, total_ll, torch.tensor(-1e9, dtype=torch.float32))

        # Vectorized MCMC in unconstrained space with sigmoid mapping
        current = x_obs_t.mean(dim=0).unsqueeze(0).repeat(50, 1) + 0.1 * torch.randn(50, d)
        current = current.clamp(lo + 0.1, hi - 0.1)
        current_lp = log_joint(current)
        step_size = 0.05 / np.sqrt(n_obs)
        burn_in = 500
        thin = 2
        total_steps = burn_in + (num_samples // 50 + 1) * thin
        samples_list = []
        for step in range(total_steps):
            prop = current + step_size * torch.randn_like(current)
            prop_lp = log_joint(prop)
            alpha = torch.clamp(torch.exp(prop_lp - current_lp), max=1.0)
            acc = torch.rand(50) < alpha
            current = torch.where(acc.unsqueeze(-1), prop, current)
            current_lp = torch.where(acc, prop_lp, current_lp)
            if step >= burn_in and (step - burn_in) % thin == 0:
                samples_list.append(current.clone())

        samples = torch.cat(samples_list, dim=0)[:num_samples]
        return samples
