"""Simulator for the Spike-and-Slab mixture model."""

import math
from typing import override

import numpy as np
import pyro.distributions as pdist
import torch
from scipy import special as sp_special
from scipy import stats

from ..base import Simulator
from ..sampling import log_normal_box_prob, rejection_sample_box


class SpikeSlab(Simulator):
    """Gaussian-mixture Spike-and-Slab likelihood under ``Uniform([-10, 10]^D)`` prior."""

    _PRIOR_LOW = -10.0
    _PRIOR_HIGH = 10.0

    def __init__(
        self,
        seed: int = 42,
        dim_parameters: int = 2,
        dim_data: int | None = None,
        **kwargs,
    ):
        if dim_data is None:
            dim_data = dim_parameters
        super().__init__(
            seed=seed, dim_parameters=dim_parameters, dim_data=dim_data, **kwargs
        )

    @override
    def get_prior(self) -> pdist.Distribution:
        return pdist.Uniform(
            self._PRIOR_LOW * torch.ones(self.dim_parameters),
            self._PRIOR_HIGH * torch.ones(self.dim_parameters),
        ).to_event(1)

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        params, is_1d = self._prepare_parameters(parameters)
        batch_size = params.shape[0]
        choose_broad = torch.rand((batch_size, 1), device=params.device) < 0.5
        std = torch.where(choose_broad, 1.0, 0.1)
        out = params + std * torch.randn_like(params)
        return out.squeeze(0) if is_1d else out

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the exact posterior under ``Uniform([-10, 10]^D)``."""
        if x_obs is None:
            x_obs = torch.zeros(self.dim_data, dtype=torch.float32)

        x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_obs_t.ndim == 1:
            x_obs_t = x_obs_t.unsqueeze(0)

        n_obs, d = x_obs_t.shape
        lo, hi = self._PRIOR_LOW, self._PRIOR_HIGH

        if n_obs == 1:
            x_single = x_obs_t[0]

            def propose(batch_size: int) -> torch.Tensor:
                choose_broad = (
                    torch.rand((batch_size, 1), device=x_single.device) < 0.5
                )
                stds = torch.where(choose_broad, 1.0, 0.1)
                return x_single.unsqueeze(0) + stds * torch.randn(
                    (batch_size, d), dtype=torch.float32, device=x_single.device
                )

            def fallback(remaining: int) -> torch.Tensor:
                x_np = x_single.cpu().numpy()
                lp_broad = float(log_normal_box_prob(x_np, 1.0, lo, hi).sum())
                lp_narrow = float(log_normal_box_prob(x_np, 0.1, lo, hi).sum())
                p_broad = float(sp_special.expit(lp_broad - lp_narrow))
                choose_broad = self._np_rng.random((remaining, 1)) < p_broad
                sigmas = np.where(choose_broad, 1.0, 0.1)
                means = np.broadcast_to(x_np[None, :], (remaining, d))
                rvs = stats.truncnorm.rvs(
                    (lo - means) / sigmas,
                    (hi - means) / sigmas,
                    loc=means,
                    scale=sigmas,
                    random_state=self._np_rng,
                )
                return torch.as_tensor(rvs, dtype=torch.float32)

            return rejection_sample_box(
                propose,
                num_samples,
                low=lo,
                high=hi,
                max_attempts=50,
                fallback_fn=fallback,
            )

        # For n_obs > 1, evaluate exact product-of-mixtures log-likelihood via parallel MH
        log_norm1 = d * math.log(2.0 * math.pi)
        log_norm2 = d * math.log(2.0 * math.pi * 0.01)
        log_half = math.log(0.5)
        neg_inf = torch.tensor(-float("inf"), dtype=torch.float32)

        def log_joint(th: torch.Tensor) -> torch.Tensor:
            diff = x_obs_t.unsqueeze(0) - th.unsqueeze(1)  # (K, n_obs, D)
            dist_sq = diff.square().sum(dim=-1)  # (K, n_obs)
            ll1 = -0.5 * (log_norm1 + dist_sq)
            ll2 = -0.5 * (log_norm2 + dist_sq / 0.01)
            ll_comp = torch.stack([ll1, ll2], dim=-1)  # (K, n_obs, 2)
            total_ll = (torch.logsumexp(ll_comp, dim=-1) + log_half).sum(dim=-1)
            in_prior = ((th >= lo) & (th <= hi)).all(dim=-1)
            return torch.where(in_prior, total_ll, neg_inf)

        # Vectorized random-walk MH with hard box prior bounds [-10, 10]^D
        n_chains = 50
        current = x_obs_t.mean(dim=0).unsqueeze(0).repeat(n_chains, 1) + 0.1 * torch.randn(
            n_chains, d
        )
        current = current.clamp(lo + 0.1, hi - 0.1)
        current_lp = log_joint(current)
        step_size = 0.05 / math.sqrt(n_obs)
        burn_in = 500
        thin = 2
        total_steps = burn_in + (num_samples // n_chains + 1) * thin
        samples_list: list[torch.Tensor] = []

        for step in range(total_steps):
            prop = current + step_size * torch.randn_like(current)
            prop_lp = log_joint(prop)
            log_u = torch.log(torch.rand(n_chains))
            acc = torch.isfinite(prop_lp) & (log_u < prop_lp - current_lp)
            current = torch.where(acc.unsqueeze(-1), prop, current)
            current_lp = torch.where(acc, prop_lp, current_lp)
            if step >= burn_in and (step - burn_in) % thin == 0:
                samples_list.append(current.clone())

        return torch.cat(samples_list, dim=0)[:num_samples]
