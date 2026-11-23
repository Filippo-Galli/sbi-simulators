"""Simulator for the Gaussian-Gaussian model.
reference: 'Compositional Score Modeling for Simulation-Based Inference'
"""

import math
from dataclasses import dataclass
from typing import override

import numpy as np
import pyro.distributions as pdist
import torch

from ..base import Simulator


@dataclass(frozen=True)
class GaussianGaussianPreset:
    """Declarative configuration for a Gaussian-Gaussian benchmark variant."""

    prior_var: float
    lik_var_range: tuple[float, float]


GAUSSIAN_GAUSSIAN_PRESETS: dict[str, GaussianGaussianPreset] = {
    "default": GaussianGaussianPreset(prior_var=1.0, lik_var_range=(0.6, 1.4)),
    "sbibm": GaussianGaussianPreset(prior_var=0.1, lik_var_range=(0.1, 0.1)),
}


class GaussianGaussian(Simulator):
    """Simulator for the Gaussian-Gaussian / Gaussian Linear model.

    Presets may be passed as a registered name (``"default"``, ``"sbibm"``) or
    directly as a :class:`GaussianGaussianPreset` data object.
    """

    def __init__(
        self,
        dim_parameters: int = 10,
        seed: int = 42,
        dim_data: int | None = None,
        preset: str | GaussianGaussianPreset = "default",
        prior_var: float | None = None,
        lik_var_range: tuple[float, float] | None = None,
        **kwargs,
    ):
        if isinstance(preset, GaussianGaussianPreset):
            cfg = preset
            self.preset = "custom"
        else:
            preset_key = preset.lower()
            if preset_key not in GAUSSIAN_GAUSSIAN_PRESETS:
                raise ValueError(
                    f"Unknown preset '{preset}'. Supported presets: {', '.join(GAUSSIAN_GAUSSIAN_PRESETS)}."
                )
            cfg = GAUSSIAN_GAUSSIAN_PRESETS[preset_key]
            self.preset = preset_key
        self.preset_config = cfg

        self.prior_var = float(cfg.prior_var if prior_var is None else prior_var)
        if self.prior_var <= 0.0:
            raise ValueError(f"prior_var must be positive, got {self.prior_var}.")

        raw_range = cfg.lik_var_range if lik_var_range is None else lik_var_range
        lo_var, hi_var = float(raw_range[0]), float(raw_range[1])
        if lo_var <= 0.0 or hi_var < lo_var:
            raise ValueError(
                f"lik_var_range must satisfy 0 < low <= high, got {raw_range}."
            )
        self.lik_var_range = (lo_var, hi_var)

        if dim_data is None:
            dim_data = dim_parameters
        super().__init__(
            seed=seed, dim_parameters=dim_parameters, dim_data=dim_data, **kwargs
        )

    def _diag_sigma(self, dim: int) -> torch.Tensor:
        lo, hi = self.lik_var_range
        if dim == 1 or lo == hi:
            return torch.full((dim,), lo, dtype=torch.float32)
        return torch.linspace(lo, hi, steps=dim, dtype=torch.float32)

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        params, is_1d = self._prepare_parameters(parameters)
        dim = params.shape[-1]
        std = torch.sqrt(self._diag_sigma(dim).to(params.device))
        out = params + torch.randn_like(params) * std
        return out.squeeze(0) if is_1d else out

    @override
    def get_prior(self) -> pdist.Distribution:
        std = math.sqrt(self.prior_var)
        return pdist.Normal(
            torch.zeros(self.dim_parameters),
            std * torch.ones(self.dim_parameters),
        ).to_event(1)

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the exact closed-form diagonal Gaussian posterior in ``O(D)`` time."""
        if x_obs is None:
            x_obs_t = torch.zeros((1, self.dim_data), dtype=torch.float32)
        else:
            x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32)
            if x_obs_t.ndim == 1:
                x_obs_t = x_obs_t.unsqueeze(0)

        n_obs, dim = x_obs_t.shape
        diag_sigma = self._diag_sigma(dim).to(x_obs_t.device)

        post_var = 1.0 / (1.0 / self.prior_var + n_obs / diag_sigma)
        post_mean = post_var * (x_obs_t.sum(dim=0) / diag_sigma)
        return post_mean + torch.sqrt(post_var) * torch.randn(
            (num_samples, dim), dtype=torch.float32, device=x_obs_t.device
        )
