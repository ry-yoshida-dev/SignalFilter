"""Abstract infinite impulse response filter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import override

import numpy as np
from scipy import signal

from .._parameters import require_integer
from ..array_types import ComplexArray, FloatArray
from ..frequency_selective_filter import FrequencySelectiveFilter


@dataclass(frozen=True, kw_only=True)
class IIRFilter(FrequencySelectiveFilter, ABC):
    """
    Recursive filter, run as a cascade of second-order sections.

    Second-order sections keep high orders and low cutoffs numerically stable,
    where a single transfer-function polynomial would not be. A ``Phase.CAUSAL``
    run starts in the steady state of the first sample, so a signal with an
    offset does not set off a start-up transient.
    """

    @property
    def second_order_sections(self) -> FloatArray:
        """
        Get the designed coefficients.

        Returns
        -------
        FloatArray
            A copy with shape ``(n_sections, 6)``; each row is
            ``[b0, b1, b2, a0, a1, a2]`` of one section.
        """
        return self._sections.copy()

    @staticmethod
    def _require_order(order: int) -> None:
        """
        Check that a design order is a positive integer.

        Raises
        ------
        TypeError
            If it is not an integer.
        ValueError
            If it is smaller than 1.
        """
        require_integer(order, "order")
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")

    @cached_property
    def _sections(self) -> FloatArray:
        return self._design()

    @abstractmethod
    def _design(self) -> FloatArray:
        """
        Design the filter.

        Returns
        -------
        FloatArray
            Second-order sections with shape ``(n_sections, 6)``.
        """

    @property
    @override
    def _padding_length(self) -> int:
        return 3 * (2 * int(self._sections.shape[0]) + 1)

    @override
    def _forward(self, samples: FloatArray) -> FloatArray:
        steady_state: FloatArray = signal.sosfilt_zi(self._sections)
        channel_axes = (np.newaxis,) * (samples.ndim - 1)
        # Scale the unit-step steady state by the first sample of each channel.
        initial: FloatArray = (
            steady_state[(..., *channel_axes)] * samples[0][np.newaxis, np.newaxis]
        )
        filtered, _ = signal.sosfilt(self._sections, samples, axis=0, zi=initial)
        return np.asarray(filtered, dtype=np.float64)

    @override
    def _forward_backward(self, samples: FloatArray) -> FloatArray:
        filtered = signal.sosfiltfilt(
            self._sections, samples, axis=0, padtype="odd", padlen=self._padding_length
        )
        return np.asarray(filtered, dtype=np.float64)

    @override
    def _transfer(self, frequencies: FloatArray) -> ComplexArray:
        _, response = signal.sosfreqz(self._sections, worN=frequencies, fs=self.sampling_rate)
        return np.asarray(response, dtype=np.complex128)
