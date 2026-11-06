"""Privacy-safe reference implementation of the JITAI-Twin pipeline."""

from .calibration import build_linear_guidance
from .data import Cohort, TrajectoryScaler
from .diffusion import GaussianDiffusion
from .model import CausalConditionalDenoiser
from .synthetic import SyntheticConfig, make_synthetic_cohorts

__all__ = [
    "CausalConditionalDenoiser",
    "Cohort",
    "GaussianDiffusion",
    "SyntheticConfig",
    "TrajectoryScaler",
    "build_linear_guidance",
    "make_synthetic_cohorts",
]

__version__ = "0.1.0"
