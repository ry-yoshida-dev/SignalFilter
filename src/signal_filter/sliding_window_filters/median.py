from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import ndimage

from ..array_types import FloatArray
from ..sliding_window_filter import SlidingWindowFilter


@dataclass(frozen=True, kw_only=True)
class MedianFilter(SlidingWindowFilter):
    """
    Median of the window centered on each sample.

    Removes impulses narrower than half the window while keeping steps sharp;
    every sample is replaced, including those that were not outliers.

    Attributes
    ----------
    window_length : int
        The number of samples the median is taken over, odd.
    boundary : Boundary
        The values assumed beyond the ends of the signal.
    """

    @override
    def _filter(self, samples: FloatArray) -> FloatArray:
        size = (self.window_length,) + (1,) * (samples.ndim - 1)
        filtered = ndimage.median_filter(self._pad(samples), size=size)
        return self._crop(np.asarray(filtered, dtype=np.float64))
