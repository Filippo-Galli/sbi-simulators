"""Simulator for the Two Moons task."""

import math
from typing import override

import numpy as np
import pyro.distributions as pdist
import torch

from ..Simulator import Simulator


class TwoMoons(Simulator):
    def __init__(self, dim_parameters: int = 2, seed: int = 42, dim_data: int = 2):
        """Initialize the TwoMoons model."""
        self.a_low = -math.pi / 2.0
        self.a_high = +math.pi / 2.0
        self.base_offset = 0.25
        self.r_loc = 0.1
        self.r_scale = 0.01

        super().__init__(seed, dim_parameters, dim_data)

        prior_bound = 1.0
        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)

    @override
    def get_prior(self) -> pdist.Distribution:
        return self.prior_dist

    @override
    def simulate(self, parameters: torch.Tensor) -> torch.Tensor:
        """Simulate one observation from parameters.

        Args:
            parameters (torch.Tensor): 2D tensor, parameters for the simulation

        Returns:
            x (torch.Tensor): Simulated observation of shape (batch_size, 2)
        """
        parameters = torch.atleast_2d(torch.as_tensor(parameters, dtype=torch.float32))
        num_samples = parameters.shape[0]

        a_dist = torch.distributions.Uniform(low=self.a_low, high=self.a_high)
        a = a_dist.sample((num_samples, 1))

        r_dist = torch.distributions.Normal(self.r_loc, self.r_scale)
        r = r_dist.sample((num_samples, 1))

        p = torch.cat(
            (
                torch.cos(a) * r + self.base_offset,
                torch.sin(a) * r,
            ),
            dim=1,
        )

        ang = torch.tensor([-math.pi / 4.0])
        c = torch.cos(ang)
        s = torch.sin(ang)
        z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
        z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)

        return p + torch.cat((-torch.abs(z0), z1), dim=1)

    def posterior_samples(
        self,
        x_obs: torch.Tensor | np.ndarray | None = None,
        num_samples: int = 1000,
        prior_bounds: tuple[float, float] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Exact posterior sampler p(theta | x_obs) for the Two Moons task
        by inverting the simulator geometry.

        Steps
        -----
        1. Sample latent crescent point p from the base distribution.
        2. For fixed x_obs, solve x_obs = p + [-|z0|, z1] for (z0, z1).
        3. Use random signs z0 = ±(p0 - x0_obs) with equal probability.
        4. Rotate back to theta: theta = R(-ang) z.
        5. Keep only theta inside the uniform prior box [-1, 1]^2.
        """
        if (
            x_obs is None
            and "num_samples" in kwargs
            and isinstance(num_samples, (torch.Tensor, np.ndarray))
        ):
            x_obs, num_samples = num_samples, kwargs.get("num_samples", 1000)
        if x_obs is None:
            x_obs = torch.tensor([-0.6397, 0.1623], dtype=torch.float32)

        if isinstance(x_obs, torch.Tensor):
            x_obs_np = x_obs.detach().cpu().numpy().astype(float).reshape(-1)
        else:
            x_obs_np = np.asarray(x_obs, dtype=float).reshape(-1)

        if x_obs_np.shape[0] != 2:
            raise ValueError(f"Expected 2D observation, got shape {x_obs_np.shape}")
        x0, x1 = x_obs_np[0], x_obs_np[1]

        if prior_bounds is None:
            prior_bounds = (-1.0, 1.0)
        lo, hi = prior_bounds
        base_offset = self.base_offset
        r_loc, r_scale = self.r_loc, self.r_scale

        # forward rotation angle and its inverse
        ang = -np.pi / 4.0
        c_inv, s_inv = np.cos(-ang), np.sin(-ang)  # R(-ang) = inverse rotation

        out = []
        collected = 0

        while collected < num_samples:
            batch = max(4 * (num_samples - collected), 4096)

            # 1. sample from crescent base
            a = np.random.uniform(self.a_low, self.a_high, size=batch)
            r = np.random.normal(r_loc, r_scale, size=batch)

            p0 = np.cos(a) * r + base_offset
            p1 = np.sin(a) * r

            # 2. solve for z0,z1 given x_obs
            # x0_obs = p0 - |z0|  =>  |z0| = p0 - x0_obs  (must be >= 0)
            # x1_obs = p1 + z1    =>  z1 = x1_obs - p1
            q0_abs = p0 - x0
            q1 = x1 - p1

            # drop impossible latent points where |z0| would be negative
            valid = q0_abs >= 0.0
            if not np.any(valid):
                continue

            q0_abs = q0_abs[valid]
            q1 = q1[valid]

            # 3. Random sign choice for z0 to represent both moon branches equally
            signs = np.where(np.random.rand(len(q0_abs)) < 0.5, 1.0, -1.0)
            q0 = signs * q0_abs

            # 4. rotate back to theta: z = R(ang) theta => theta = R(-ang) z
            theta1 = c_inv * q0 - s_inv * q1
            theta2 = s_inv * q0 + c_inv * q1
            theta = np.stack([theta1, theta2], axis=1)

            # 5. enforce prior support (uniform box)
            keep = (
                (theta[:, 0] >= lo)
                & (theta[:, 0] <= hi)
                & (theta[:, 1] >= lo)
                & (theta[:, 1] <= hi)
            )
            theta = theta[keep]

            if theta.shape[0] > 0:
                out.append(theta)
                collected += theta.shape[0]

        samples = np.concatenate(out, axis=0)
        np.random.shuffle(samples)
        return torch.as_tensor(samples[:num_samples], dtype=torch.float32)
