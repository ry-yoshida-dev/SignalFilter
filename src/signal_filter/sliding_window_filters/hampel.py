from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import ArrayLike

from .._samples import as_samples
from ..array_types import BoolArray, FloatArray
from ..sliding_window_filter import SlidingWindowFilter

# Scales the median absolute deviation to the standard deviation of normal noise.
_NORMAL_CONSISTENCY: float = 1.4826
# Window values gathered at once, which bounds the memory of a call (8 bytes each) for any length.
_CHUNK_ELEMENTS: int = 1 << 22


@dataclass(frozen=True, kw_only=True)
class HampelFilter(SlidingWindowFilter):
    """
    Replace outliers with the median of the window centered on them.

    A sample is an outlier when it is more than ``threshold`` robust standard
    deviations (1.4826 times the median absolute deviation of its window) away
    from that window's median. Other samples are kept exactly, so unlike
    :class:`MedianFilter` the signal is not smoothed. In a perfectly flat window
    any deviation counts as an outlier.

    Attributes
    ----------
    window_length : int
        The number of samples each median is taken over, odd.
    boundary : Boundary
        The values assumed beyond the ends of the signal.
    threshold : float
        The number of robust standard deviations beyond which a sample is an
        outlier; 0 makes this a median filter.

    Raises
    ------
    TypeError
        If ``window_length`` is not an integer.
    ValueError
        If ``window_length`` is not a positive odd number, or ``threshold`` is
        negative or not finite.
    """

    threshold: float = 3.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if not (np.isfinite(self.threshold) and self.threshold >= 0.0):
            raise ValueError(f"threshold must be non-negative and finite, got {self.threshold}")

    def detect(self, signal: ArrayLike) -> BoolArray:
        """
        Find the outliers :meth:`apply` replaces.

        Parameters
        ----------
        signal : ArrayLike
            Real samples with shape ``(n_samples, *channel_shape)``.

        Returns
        -------
        BoolArray
            True at each outlier, with the shape of the signal. Each channel is
            judged on its own.

        Raises
        ------
        TypeError
            If the signal is not real-valued numbers.
        ValueError
            If the signal is 0-D, empty, or not finite.
        """
        samples = as_samples(signal, self.minimum_length)
        _, is_outlier = self._medians_and_outliers(samples)
        return is_outlier

    @override
    def _filter(self, samples: FloatArray) -> FloatArray:
        medians, is_outlier = self._medians_and_outliers(samples)
        samples[is_outlier] = medians[is_outlier]
        return samples

    def _medians_and_outliers(self, samples: FloatArray) -> tuple[FloatArray, BoolArray]:
        windows = self._windows(samples)
        medians: FloatArray = np.empty_like(samples)
        deviations: FloatArray = np.empty_like(samples)
        values_per_sample = max(1, samples[0].size) * self.window_length
        chunk_length = max(1, _CHUNK_ELEMENTS // values_per_sample)
        for start in range(0, samples.shape[0], chunk_length):
            chunk = slice(start, start + chunk_length)
            chunk_medians: FloatArray = np.median(windows[chunk], axis=-1)
            medians[chunk] = chunk_medians
            deviations[chunk] = np.median(
                np.abs(windows[chunk] - chunk_medians[..., np.newaxis]), axis=-1
            )
        is_outlier: BoolArray = np.abs(samples - medians) > (
            self.threshold * _NORMAL_CONSISTENCY * deviations
        )
        return medians, is_outlier
