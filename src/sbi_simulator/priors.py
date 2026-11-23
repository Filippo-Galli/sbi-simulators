"""Hashable prior specifications unifying Pyro distributions and JAX MCMC."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
import pyro.distributions as pdist
import torch


class PriorSpec(ABC):
    """Abstract hashable prior specification usable by both Pyro and JAX JIT."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def to_pyro(self) -> pdist.Distribution:
        """Construct the corresponding Pyro event distribution."""

    @abstractmethod
    def jax_log_prob_unconstrained(self, z: jax.Array) -> jax.Array:
        """Evaluate log prior density at unconstrained point ``z`` in JAX."""

    @abstractmethod
    def jax_sample_unconstrained(self, key: jax.Array, count: int) -> jax.Array:
        """Draw ``count`` unconstrained samples of shape ``(count, dim)`` in JAX."""

    @abstractmethod
    def canonical_point(self) -> tuple[float, ...]:
        """Return a representative point inside the prior support."""

    def to_unconstrained(self, theta: jax.Array) -> jax.Array:
        """Map constrained parameter ``theta`` into unconstrained sampler space."""
        return theta

    def from_unconstrained(self, z: jax.Array) -> jax.Array:
        """Map unconstrained sampler coordinate ``z`` back to parameter space."""
        return z

    def clip_unconstrained(self, z: jax.Array) -> jax.Array:
        """Project unconstrained state ``z`` onto valid support when bounded."""
        return z

    def jax_log_prob(self, theta: jax.Array) -> jax.Array:
        """Evaluate log prior density at parameter-space point ``theta`` in JAX."""
        return self.jax_log_prob_unconstrained(theta)


@dataclass(frozen=True)
class NormalPrior(PriorSpec):
    """Independent diagonal Normal prior ``N(loc, diag(scale^2))``."""

    loc: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.loc) == 0 or len(self.loc) != len(self.scale):
            raise ValueError(
                f"NormalPrior loc and scale must be non-empty and match in length, "
                f"got {len(self.loc)} and {len(self.scale)}."
            )
        if any(s <= 0.0 for s in self.scale):
            raise ValueError("NormalPrior scales must be strictly positive.")

    @classmethod
    def isotropic(
        cls, dim: int, loc: float = 0.0, scale: float = 1.0
    ) -> NormalPrior:
        """Construct an isotropic ``dim``-dimensional Normal prior."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        return cls(loc=(float(loc),) * dim, scale=(float(scale),) * dim)

    @property
    @override
    def dim(self) -> int:
        return len(self.loc)

    @override
    def to_pyro(self) -> pdist.Distribution:
        return pdist.Normal(
            torch.tensor(self.loc, dtype=torch.float32),
            torch.tensor(self.scale, dtype=torch.float32),
        ).to_event(1)

    @override
    def jax_log_prob_unconstrained(self, z: jax.Array) -> jax.Array:
        loc_arr = jnp.asarray(self.loc, dtype=jnp.float32)
        scale_arr = jnp.asarray(self.scale, dtype=jnp.float32)
        log_norm = jnp.sum(jnp.log(scale_arr)) + 0.5 * self.dim * jnp.log(2.0 * jnp.pi)
        return -0.5 * jnp.sum(((z - loc_arr) / scale_arr) ** 2) - log_norm

    @override
    def jax_sample_unconstrained(self, key: jax.Array, count: int) -> jax.Array:
        loc_arr = jnp.asarray(self.loc, dtype=jnp.float32)
        scale_arr = jnp.asarray(self.scale, dtype=jnp.float32)
        return loc_arr + jax.random.normal(key, shape=(count, self.dim)) * scale_arr

    @override
    def canonical_point(self) -> tuple[float, ...]:
        return self.loc


@dataclass(frozen=True)
class UniformPrior(PriorSpec):
    """Independent box Uniform prior ``Uniform(low, high)``."""

    low: tuple[float, ...]
    high: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.low) == 0 or len(self.low) != len(self.high):
            raise ValueError(
                f"UniformPrior low and high must be non-empty and match in length, "
                f"got {len(self.low)} and {len(self.high)}."
            )
        if any(lo >= hi for lo, hi in zip(self.low, self.high, strict=True)):
            raise ValueError("UniformPrior requires low < high elementwise.")

    @classmethod
    def box(cls, dim: int, low: float, high: float) -> UniformPrior:
        """Construct a hypercube ``Uniform([low, high]^dim)`` prior."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        return cls(low=(float(low),) * dim, high=(float(high),) * dim)

    @property
    @override
    def dim(self) -> int:
        return len(self.low)

    @override
    def to_pyro(self) -> pdist.Distribution:
        return pdist.Uniform(
            torch.tensor(self.low, dtype=torch.float32),
            torch.tensor(self.high, dtype=torch.float32),
        ).to_event(1)

    @override
    def jax_log_prob_unconstrained(self, z: jax.Array) -> jax.Array:
        low_arr = jnp.asarray(self.low, dtype=jnp.float32)
        high_arr = jnp.asarray(self.high, dtype=jnp.float32)
        in_bounds = jnp.all((z >= low_arr) & (z <= high_arr))
        log_density = -jnp.sum(jnp.log(high_arr - low_arr))
        return jnp.where(in_bounds, log_density, -jnp.inf)

    @override
    def jax_sample_unconstrained(self, key: jax.Array, count: int) -> jax.Array:
        low_arr = jnp.asarray(self.low, dtype=jnp.float32)
        high_arr = jnp.asarray(self.high, dtype=jnp.float32)
        return jax.random.uniform(
            key, shape=(count, self.dim), minval=low_arr, maxval=high_arr
        )

    @override
    def clip_unconstrained(self, z: jax.Array) -> jax.Array:
        low_arr = jnp.asarray(self.low, dtype=jnp.float32)
        high_arr = jnp.asarray(self.high, dtype=jnp.float32)
        return jnp.clip(z, low_arr, high_arr)

    @override
    def canonical_point(self) -> tuple[float, ...]:
        return tuple(
            0.5 * (lo + hi) for lo, hi in zip(self.low, self.high, strict=True)
        )


@dataclass(frozen=True)
class LogNormalPrior(PriorSpec):
    """Independent diagonal LogNormal prior sampled in unconstrained log-space."""

    loc: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.loc) == 0 or len(self.loc) != len(self.scale):
            raise ValueError(
                f"LogNormalPrior loc and scale must be non-empty and match in length, "
                f"got {len(self.loc)} and {len(self.scale)}."
            )
        if any(s <= 0.0 for s in self.scale):
            raise ValueError("LogNormalPrior scales must be strictly positive.")

    @property
    @override
    def dim(self) -> int:
        return len(self.loc)

    @override
    def to_pyro(self) -> pdist.Distribution:
        return pdist.LogNormal(
            torch.tensor(self.loc, dtype=torch.float32),
            torch.tensor(self.scale, dtype=torch.float32),
        ).to_event(1)

    @override
    def to_unconstrained(self, theta: jax.Array) -> jax.Array:
        return jnp.log(jnp.maximum(theta, 1e-12))

    @override
    def from_unconstrained(self, z: jax.Array) -> jax.Array:
        return jnp.exp(z)

    @override
    def jax_log_prob_unconstrained(self, z: jax.Array) -> jax.Array:
        loc_arr = jnp.asarray(self.loc, dtype=jnp.float32)
        scale_arr = jnp.asarray(self.scale, dtype=jnp.float32)
        log_norm = jnp.sum(jnp.log(scale_arr)) + 0.5 * self.dim * jnp.log(2.0 * jnp.pi)
        return -0.5 * jnp.sum(((z - loc_arr) / scale_arr) ** 2) - log_norm

    @override
    def jax_sample_unconstrained(self, key: jax.Array, count: int) -> jax.Array:
        loc_arr = jnp.asarray(self.loc, dtype=jnp.float32)
        scale_arr = jnp.asarray(self.scale, dtype=jnp.float32)
        return loc_arr + jax.random.normal(key, shape=(count, self.dim)) * scale_arr

    @override
    def jax_log_prob(self, theta: jax.Array) -> jax.Array:
        pos = jnp.all(theta > 0.0)
        z = jnp.log(jnp.maximum(theta, 1e-12))
        return jnp.where(pos, self.jax_log_prob_unconstrained(z) - jnp.sum(z), -jnp.inf)

    @override
    def canonical_point(self) -> tuple[float, ...]:
        return tuple(math.exp(m) for m in self.loc)


def _tensor_to_tuple(t: torch.Tensor, batch_shape: torch.Size) -> tuple[float, ...]:
    target_shape = batch_shape if len(batch_shape) > 0 else torch.Size((1,))
    arr = torch.broadcast_to(t.detach().cpu(), target_shape).reshape(-1).tolist()
    return tuple(float(v) for v in arr)


def as_prior_spec(prior: PriorSpec | pdist.Distribution) -> PriorSpec:
    """Convert a ``PriorSpec`` or Pyro/Torch distribution into a hashable ``PriorSpec``."""
    if isinstance(prior, PriorSpec):
        return prior

    base: torch.distributions.Distribution | pdist.Distribution = prior
    while isinstance(base, torch.distributions.Independent):
        base = base.base_dist

    if isinstance(base, torch.distributions.LogNormal):
        return LogNormalPrior(
            loc=_tensor_to_tuple(base.loc, base.batch_shape),
            scale=_tensor_to_tuple(base.scale, base.batch_shape),
        )
    if isinstance(base, torch.distributions.Normal):
        return NormalPrior(
            loc=_tensor_to_tuple(base.loc, base.batch_shape),
            scale=_tensor_to_tuple(base.scale, base.batch_shape),
        )
    if isinstance(base, torch.distributions.Uniform):
        return UniformPrior(
            low=_tensor_to_tuple(base.low, base.batch_shape),
            high=_tensor_to_tuple(base.high, base.batch_shape),
        )

    raise TypeError(
        f"Unsupported prior distribution {type(prior).__name__}. "
        f"Expected PriorSpec or Normal/Uniform/LogNormal distribution."
    )
