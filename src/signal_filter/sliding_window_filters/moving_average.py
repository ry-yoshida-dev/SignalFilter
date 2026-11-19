from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import ndimage

from ..array_types import FloatArray
from ..sliding_window_filter import SlidingWindowFilter


@dataclass(frozen=True, kw_only=True)
class MovingAverageFilter(SlidingWindowFilter):
    """
    Mean of the window centered on each sample.

    The fastest smoother and optimal for reducing white noise while keeping a
    step sharp, but it flattens peaks and passes some high frequencies through
    its side lobes.

    Attributes
    ----------
    window_length : int
        The number of samples averaged, odd.
    boundary : Boundary
        The values assumed beyond the ends of the signal.
    """

    @override
    def _filter(self, samples: FloatArray) -> FloatArray:
        filtered = ndimage.uniform_filter1d(self._pad(samples), self.window_length, axis=0)
        return self._crop(np.asarray(filtered, dtype=np.float64))
