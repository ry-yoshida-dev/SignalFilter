"""Abstract finite impulse response filter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import override

import numpy as np
from scipy import signal

from ..array_types import ComplexArray, FloatArray
from ..frequency_selective_filter import FrequencySelectiveFilter

_DENOMINATOR: FloatArray = np.ones(1)


@dataclass(frozen=True, kw_only=True)
class FIRFilter(FrequencySelectiveFilter, ABC):
    """
    Non-recursive filter with symmetric, linear-phase coefficients.

    Linear phase delays every frequency equally, so a ``Phase.CAUSAL`` run keeps
    the waveform and only shifts it by ``group_delay``. A ``Phase.CAUSAL`` run
    starts in the steady state of the first sample.
    """

    @property
    def coefficients(self) -> FloatArray:
        """
        Get the impulse response.

        Returns
        -------
        FloatArray
            A copy of the symmetric coefficients, an odd number of them.
        """
        return self._coefficients.copy()

    @property
    def group_delay(self) -> float:
        """
        Get the delay a ``Phase.CAUSAL`` run shifts the signal by.

        Returns
        -------
        float
            Half the filter length minus one sample, in seconds.
        """
        return 0.5 * (int(self._coefficients.shape[0]) - 1) / self.sampling_rate

    @cached_property
    def _coefficients(self) -> FloatArray:
        return self._design()

    @abstractmethod
    def _design(self) -> FloatArray:
        """
        Design the filter.

        Returns
        -------
        FloatArray
            Symmetric coefficients with an odd length.
        """

    @property
    @override
    def _padding_length(self) -> int:
        return 3 * int(self._coefficients.shape[0])

    @override
    def _forward(self, samples: FloatArray) -> FloatArray:
        steady_state: FloatArray = signal.lfilter_zi(self._coefficients, _DENOMINATOR)
        channel_axes = (np.newaxis,) * (samples.ndim - 1)
        # Scale the unit-step steady state by the first sample of each channel.
        initial: FloatArray = steady_state[(..., *channel_axes)] * samples[0][np.newaxis]
        filtered, _ = signal.lfilter(self._coefficients, _DENOMINATOR, samples, axis=0, zi=initial)
        return np.asarray(filtered, dtype=np.float64)

    @override
    def _forward_backward(self, samples: FloatArray) -> FloatArray:
        filtered = signal.filtfilt(
            self._coefficients,
            _DENOMINATOR,
            samples,
            axis=0,
            padtype="odd",
            padlen=self._padding_length,
        )
        return np.asarray(filtered, dtype=np.float64)

    @override
    def _transfer(self, frequencies: FloatArray) -> ComplexArray:
        _, response = signal.freqz(self._coefficients, worN=frequencies, fs=self.sampling_rate)
        return np.asarray(response, dtype=np.complex128)
