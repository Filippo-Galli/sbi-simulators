"""sbi-simulator: Benchmark simulation models for Simulation-Based Inference."""

from collections.abc import Mapping
from importlib import import_module

from .Simulator import Simulator

__version__ = "0.1.0"

_AVAILABLE = {
    "BimodalGaussians": ".BimodalGaussians.simulator",
    "GaussianGaussian": ".GaussianGaussian.simulator",
    "SpikeSlab": ".SpikeSlab.simulator",
    "SIR": ".SIR.simulator",
    "TwoMoons": ".TwoMoons.simulator",
    "LotkaVolterra": ".LotkaVolterra.simulator",
    "UniformGaussian": ".UniformGaussian.simulator",
}


class _LazySimulators(Mapping):
    """Lazy-loading Mapping of simulator name -> simulator class.

    Imports the requested simulator module only when accessed to avoid
    heavy initialization or circular imports.
    """

    def __init__(self, mapping):
        self._mapping = dict(mapping)
        self._cache = {}

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        if key not in self._mapping:
            raise KeyError(key)
        module_path = self._mapping[key]
        module = import_module(module_path, package=__name__)
        cls = getattr(module, key)
        self._cache[key] = cls
        return cls

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def __repr__(self):
        return f"SIMULATORS({list(self._mapping.keys())})"


SIMULATORS = _LazySimulators(_AVAILABLE)


def __getattr__(name: str):
    if name in _AVAILABLE:
        return SIMULATORS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_AVAILABLE.keys()))


__all__ = ["SIMULATORS", "Simulator", *list(_AVAILABLE.keys())]
