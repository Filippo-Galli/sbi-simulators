"""Simulator for the Bimodal Gaussians mixture model."""

import math
from typing import override

import numpy as np
import torch
from pyro import distributions as pdist
from scipy import special as sp_special
from scipy import stats

from ..base import Simulator
from ..sampling import log_normal_box_prob, rejection_sample_box


class BimodalGaussians(Simulator):
    """Symmetric bimodal Gaussian likelihood under ``Uniform([-10, 10]^d)`` prior."""

    _PRIOR_LOW = -10.0
    _PRIOR_HIGH = 10.0

    def __init__(
        self,
        dim_parameters: int = 2,
        seed: int = 42,
        sigma: float = 0.1,
        dim_data: int | None = None,
        **kwargs,
    ):
        self.sigma = float(sigma)
        self.shift = 10.0 * self.sigma
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
        signs = torch.where(
            torch.rand((batch_size, 1), device=params.device) < 0.5, 1.0, -1.0
        )
        out = params + signs * self.shift + self.sigma * torch.randn_like(params)
        return out.squeeze(0) if is_1d else out

    def _sample_trunc_box(self, means_np: np.ndarray, sigma_eff: float) -> torch.Tensor:
        a = (self._PRIOR_LOW - means_np) / sigma_eff
        b = (self._PRIOR_HIGH - means_np) / sigma_eff
        draws = stats.truncnorm.rvs(
            a,
            b,
            loc=means_np,
            scale=sigma_eff,
            random_state=self._np_rng,
        )
        return torch.as_tensor(draws, dtype=torch.float32)

    def _exact_marginal_log_likelihood(
        self, theta: np.ndarray, x_np: np.ndarray
    ) -> np.ndarray:
        """Compute exact log p(x_{1:N} | theta) = sum_i log(0.5 N(x_i|th+s) + 0.5 N(x_i|th-s))."""
        # theta: (C, D), x_np: (N, D)
        diff = x_np[None, :, :] - theta[:, None, :]
        sq_pos = np.sum((diff - self.shift) ** 2, axis=-1)
        sq_neg = np.sum((diff + self.shift) ** 2, axis=-1)
        inv_2var = -0.5 / (self.sigma**2)
        lp_pos = sq_pos * inv_2var
        lp_neg = sq_neg * inv_2var
        # Constant -0.5 * d * log(2*pi*sigma^2) - log(2) cancels in MH ratio
        return np.sum(np.logaddexp(lp_pos, lp_neg), axis=-1)

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the exact posterior ``p(theta | x_1, ..., x_N)`` under ``Uniform([-10, 10]^d)``.

        - ``N == 1``: exact 2-component truncated Gaussian mixture via rejection sampling
          with exact ``truncnorm`` fallback.
        - ``2 <= N <= 12``: exact finite-mixture enumeration over all ``2^N`` latent sign
          assignments with exact truncated-box normalizing constants.
        - ``N > 12``: multi-chain data-augmented Gibbs sampler coupled with symmetric
          exact-marginal mode-jump Metropolis-Hastings moves.
        """
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
                signs = torch.where(
                    torch.rand((batch_size, 1), device=x_single.device) < 0.5, 1.0, -1.0
                )
                means = x_single.unsqueeze(0) - signs * self.shift
                return means + self.sigma * torch.randn(
                    (batch_size, d), dtype=torch.float32, device=x_single.device
                )

            def fallback(remaining: int) -> torch.Tensor:
                x_np = x_single.cpu().numpy()
                mu_pos = x_np - self.shift
                mu_neg = x_np + self.shift
                lp_pos = float(
                    log_normal_box_prob(mu_pos, self.sigma, lo, hi).sum()
                )
                lp_neg = float(
                    log_normal_box_prob(mu_neg, self.sigma, lo, hi).sum()
                )
                p_pos = float(sp_special.expit(lp_pos - lp_neg))
                choose_pos = self._np_rng.random(remaining) < p_pos
                means = np.where(choose_pos[:, None], mu_pos, mu_neg)
                return self._sample_trunc_box(means, self.sigma)

            return rejection_sample_box(
                propose,
                num_samples,
                low=lo,
                high=hi,
                max_attempts=50,
                fallback_fn=fallback,
            )

        x_np = x_obs_t.detach().cpu().numpy().astype(np.float64)
        x_mean = x_np.mean(axis=0)
        sigma_eff = self.sigma / math.sqrt(n_obs)

        if n_obs <= 12:
            # Exact enumeration of all 2^N sign vectors s in {-1, +1}^N
            n_configs = 1 << n_obs
            bits = (np.arange(n_configs)[:, None] >> np.arange(n_obs)) & 1
            signs = np.where(bits == 1, 1.0, -1.0)  # (2^N, N)
            s_bar = signs.mean(axis=1)  # (2^N,)

            # Algebraic identity: SSE(s) = const - 2*shift*(S @ c) + N*D*shift^2*(1 - s_bar^2)
            c = (x_np - x_mean).sum(axis=1)  # (N,)
            sse_rel = -2.0 * self.shift * (signs @ c) + (
                n_obs * d * (self.shift**2) * (1.0 - s_bar**2)
            )
            means_m = x_mean[None, :] - s_bar[:, None] * self.shift  # (2^N, D)
            log_box = log_normal_box_prob(means_m, sigma_eff, lo, hi).sum(axis=1)
            log_w = -sse_rel / (2.0 * (self.sigma**2)) + log_box
            probs = np.exp(log_w - log_w.max())
            probs /= probs.sum()

            idx = self._np_rng.choice(n_configs, size=num_samples, p=probs)
            return self._sample_trunc_box(means_m[idx], sigma_eff)

        # For N > 12: Data-Augmented Gibbs + symmetric exact-marginal mode-jump MH
        n_chains = 50
        burn_in = 150
        thin = 2
        steps_needed = math.ceil(num_samples / n_chains) * thin
        total_steps = burn_in + steps_needed

        alphas = np.linspace(-1.0, 1.0, n_chains)
        theta = np.clip(x_mean[None, :] - alphas[:, None] * self.shift, lo, hi)
        inv_var = 1.0 / (self.sigma**2)
        jump_deltas = np.array([-2.0, -1.0, 1.0, 2.0]) * (2.0 / n_obs)
        jump_deltas = np.concatenate([jump_deltas, np.array([-2.0, 2.0])])

        samples_out: list[np.ndarray] = []
        for step in range(total_steps):
            # 1. Exact conditional draw: s_i | theta, x_i
            logits = (2.0 * self.shift * inv_var) * (
                x_np[None, :, :] - theta[:, None, :]
            ).sum(axis=-1)
            p_pos = sp_special.expit(logits)
            s = np.where(self._np_rng.random(logits.shape) < p_pos, 1.0, -1.0)

            # 2. Exact conditional draw: theta | s, x_{1:N}
            y_bar = x_mean[None, :] - s.mean(axis=1)[:, None] * self.shift
            a = (lo - y_bar) / sigma_eff
            b = (hi - y_bar) / sigma_eff
            theta = stats.truncnorm.rvs(
                a, b, loc=y_bar, scale=sigma_eff, random_state=self._np_rng
            )

            # 3. Symmetric exact-marginal mode-jump MH along 1_d
            delta = self._np_rng.choice(jump_deltas, size=(n_chains, 1))
            prop = theta + delta * self.shift
            in_box = np.all((prop >= lo) & (prop <= hi), axis=-1)
            curr_ll = self._exact_marginal_log_likelihood(theta, x_np)
            prop_ll = self._exact_marginal_log_likelihood(
                np.clip(prop, lo, hi), x_np
            )
            log_u = np.log(self._np_rng.random(n_chains))
            accept = in_box & (log_u < prop_ll - curr_ll)
            theta = np.where(accept[:, None], prop, theta)

            if step >= burn_in and (step - burn_in) % thin == 0:
                samples_out.append(theta.copy())

        stacked = np.concatenate(samples_out, axis=0)
        self._np_rng.shuffle(stacked)
        return torch.as_tensor(stacked[:num_samples], dtype=torch.float32)
