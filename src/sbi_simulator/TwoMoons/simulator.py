"""Simulator for the Two Moons task."""

import math
from typing import override

import numpy as np
import pyro.distributions as pdist
import torch

from ..base import Simulator

_ANG = -math.pi / 4.0
_COS_ANG = math.cos(_ANG)
_SIN_ANG = math.sin(_ANG)
_COS_INV = _COS_ANG  # cos(-ang) == cos(ang)
_SIN_INV = -_SIN_ANG  # sin(-ang) == -sin(ang)


class TwoMoons(Simulator):
    """Benchmark Two Moons simulator with multimodal crescent posterior."""

    def __init__(
        self,
        dim_parameters: int = 2,
        seed: int = 42,
        dim_data: int = 2,
        **kwargs,
    ):
        """Initialize the TwoMoons model."""
        if dim_parameters != 2:
            raise ValueError(
                f"TwoMoons requires dim_parameters=2, got {dim_parameters}."
            )
        self.a_low = -math.pi / 2.0
        self.a_high = +math.pi / 2.0
        self.base_offset = 0.25
        self.r_loc = 0.1
        self.r_scale = 0.01

        super().__init__(seed, dim_parameters, dim_data, **kwargs)

        prior_bound = 1.0
        self.prior_dist = pdist.Uniform(
            low=-prior_bound * torch.ones((self.dim_parameters,)),
            high=+prior_bound * torch.ones((self.dim_parameters,)),
        ).to_event(1)

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    @override
    def simulate(self, parameters: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Simulate observations from parameters of shape ``(2,)`` or ``(B, 2)``."""
        params, is_1d = self._prepare_parameters(parameters)
        if params.shape[-1] != 2:
            raise ValueError(f"Expected 2 parameters, got {params.shape[-1]}.")

        batch_size = params.shape[0]
        device = params.device

        a = torch.empty((batch_size, 1), dtype=torch.float32, device=device).uniform_(
            self.a_low, self.a_high
        )
        r = self.r_loc + self.r_scale * torch.randn(
            (batch_size, 1), dtype=torch.float32, device=device
        )

        p = torch.cat(
            (torch.cos(a) * r + self.base_offset, torch.sin(a) * r),
            dim=1,
        )

        z0 = _COS_ANG * params[:, 0:1] - _SIN_ANG * params[:, 1:2]
        z1 = _SIN_ANG * params[:, 0:1] + _COS_ANG * params[:, 1:2]
        out = p + torch.cat((-torch.abs(z0), z1), dim=1)
        return out.squeeze(0) if is_1d else out

    @override
    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        prior_bounds: tuple[float, float] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Exact posterior sampler ``p(theta | x_obs)`` by inverting the crescent geometry."""
        if x_obs is None:
            x_obs = torch.tensor([-0.6397, 0.1623], dtype=torch.float32)

        x_obs_flat = torch.as_tensor(x_obs, dtype=torch.float32).reshape(-1)
        if x_obs_flat.shape[0] != 2:
            raise ValueError(f"Expected 2D observation, got shape {x_obs_flat.shape}")
        x0 = float(x_obs_flat[0].item())
        x1 = float(x_obs_flat[1].item())

        lo, hi = (-1.0, 1.0) if prior_bounds is None else prior_bounds

        out: list[torch.Tensor] = []
        collected = 0
        max_attempts = 200

        for _ in range(max_attempts):
            if collected >= num_samples:
                break
            batch = max(4 * (num_samples - collected), 4096)

            a = torch.empty(batch, dtype=torch.float32).uniform_(
                self.a_low, self.a_high
            )
            r = self.r_loc + self.r_scale * torch.randn(batch, dtype=torch.float32)

            p0 = torch.cos(a) * r + self.base_offset
            p1 = torch.sin(a) * r

            q0_abs = p0 - x0
            q1 = x1 - p1

            valid = q0_abs >= 0.0
            if not valid.any():
                continue

            q0_abs = q0_abs[valid]
            q1 = q1[valid]

            signs = torch.where(torch.rand(q0_abs.shape[0]) < 0.5, 1.0, -1.0)
            q0 = signs * q0_abs

            theta1 = _COS_INV * q0 - _SIN_INV * q1
            theta2 = _SIN_INV * q0 + _COS_INV * q1
            theta = torch.stack([theta1, theta2], dim=1)

            keep = (
                (theta[:, 0] >= lo)
                & (theta[:, 0] <= hi)
                & (theta[:, 1] >= lo)
                & (theta[:, 1] <= hi)
            )
            accepted = theta[keep]
            if accepted.shape[0] > 0:
                out.append(accepted)
                collected += accepted.shape[0]

        if collected < num_samples:
            raise RuntimeError(
                f"TwoMoons posterior sampling failed to collect {num_samples} samples "
                f"within [{lo}, {hi}] after {max_attempts} attempts (collected {collected})."
            )

        samples = torch.cat(out, dim=0)
        perm = torch.randperm(samples.shape[0])
        return samples[perm[:num_samples]]
