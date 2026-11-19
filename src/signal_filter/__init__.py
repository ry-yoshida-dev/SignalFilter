"""Typed digital filters for signal analysis."""

from .array_types import BoolArray, ComplexArray, FloatArray
from .band import Band
from .band_type import BandType
from .boundary import Boundary
from .frequency_response import FrequencyResponse
from .frequency_selective_filter import FrequencySelectiveFilter
from .frequency_selective_filters import (
    BesselFilter,
    BesselNormalization,
    ButterworthFilter,
    ChebyshevType1Filter,
    ChebyshevType2Filter,
    EllipticFilter,
    FIRFilter,
    IIRFilter,
    KaiserFIRFilter,
    NotchFilter,
    WindowFIRFilter,
    WindowFunction,
)
from .phase import Phase
from .signal_filter import SignalFilter
from .sliding_window_filter import SlidingWindowFilter
from .sliding_window_filters import (
    HampelFilter,
    MedianFilter,
    MovingAverageFilter,
    SavitzkyGolayFilter,
)

__all__ = [
    "Band",
    "BandType",
    "BesselFilter",
    "BesselNormalization",
    "BoolArray",
    "Boundary",
    "ButterworthFilter",
    "ChebyshevType1Filter",
    "ChebyshevType2Filter",
    "ComplexArray",
    "EllipticFilter",
    "FIRFilter",
    "FloatArray",
    "FrequencyResponse",
    "FrequencySelectiveFilter",
    "HampelFilter",
    "IIRFilter",
    "KaiserFIRFilter",
    "MedianFilter",
    "MovingAverageFilter",
    "NotchFilter",
    "Phase",
    "SavitzkyGolayFilter",
    "SignalFilter",
    "SlidingWindowFilter",
    "WindowFIRFilter",
    "WindowFunction",
]
