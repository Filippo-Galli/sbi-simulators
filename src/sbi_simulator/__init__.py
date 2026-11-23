"""sbi-simulator: Benchmark simulation models for Simulation-Based Inference."""

from importlib.metadata import PackageNotFoundError, version

from .AR1 import AR1
from .base import Simulator
from .BimodalGaussians import BimodalGaussians
from .compositional import extract_two_step_pairs, sample_two_step
from .DoubleWell import DoubleWell
from .dynamical import DynamicalSimulator
from .GaussianGaussian import (
    GAUSSIAN_GAUSSIAN_PRESETS,
    GaussianGaussian,
    GaussianGaussianPreset,
)
from .LinearSDE import LinearSDE
from .LotkaVolterra import LV_PRESETS, LotkaVolterra, LVPreset
from .priors import (
    LogNormalPrior,
    NormalPrior,
    PriorSpec,
    UniformPrior,
    as_prior_spec,
)
from .SIR import SIR, SIR_PRESETS, SIRPreset
from .SLCP import SLCP
from .SpikeSlab import SpikeSlab
from .TwoMoons import TwoMoons
from .UniformGaussian import (
    UNIFORM_GAUSSIAN_PRESETS,
    UniformGaussian,
    UniformGaussianPreset,
)

try:
    __version__ = version("sbi-simulator")
except PackageNotFoundError:
    __version__ = "0.3.1"

SIMULATORS = {
    "AR1": AR1,
    "BimodalGaussians": BimodalGaussians,
    "DoubleWell": DoubleWell,
    "GaussianGaussian": GaussianGaussian,
    "LinearSDE": LinearSDE,
    "LotkaVolterra": LotkaVolterra,
    "SIR": SIR,
    "SLCP": SLCP,
    "SpikeSlab": SpikeSlab,
    "TwoMoons": TwoMoons,
    "UniformGaussian": UniformGaussian,
}

__all__ = [
    "AR1",
    "GAUSSIAN_GAUSSIAN_PRESETS",
    "LV_PRESETS",
    "SIMULATORS",
    "SIR",
    "SIR_PRESETS",
    "SLCP",
    "UNIFORM_GAUSSIAN_PRESETS",
    "BimodalGaussians",
    "DoubleWell",
    "DynamicalSimulator",
    "GaussianGaussian",
    "GaussianGaussianPreset",
    "LVPreset",
    "LinearSDE",
    "LogNormalPrior",
    "LotkaVolterra",
    "NormalPrior",
    "PriorSpec",
    "SIRPreset",
    "Simulator",
    "SpikeSlab",
    "TwoMoons",
    "UniformGaussian",
    "UniformGaussianPreset",
    "UniformPrior",
    "__version__",
    "as_prior_spec",
    "extract_two_step_pairs",
    "sample_two_step",
]
