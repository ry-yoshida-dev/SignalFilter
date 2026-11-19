"""Abstract filter that passes or rejects frequency bands."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, override

import numpy as np
from numpy.typing import ArrayLike

from .array_types import ComplexArray, FloatArray
from .band import Band
from .band_type import BandType
from .frequency_response import FrequencyResponse
from .phase import Phase
from .signal_filter import SignalFilter

_DEFAULT_RESPONSE_POINTS: int = 512


@dataclass(frozen=True, kw_only=True)
class FrequencySelectiveFilter(SignalFilter, ABC):
    """
    Linear time-invariant filter designed in the frequency domain.

    Frequencies are given in hertz and must lie below the Nyquist frequency,
    half of ``sampling_rate``.

    Attributes
    ----------
    sampling_rate : float
        The number of samples per second.
    phase : Phase
        Whether to run the filter forward and backward (``Phase.ZERO``, no
        delay) or forward only (``Phase.CAUSAL``, real-time compatible).

    Raises
    ------
    ValueError
        If ``sampling_rate`` is not positive and finite.
    """

    sampling_rate: float
    phase: Phase = Phase.ZERO

    def __post_init__(self) -> None:
        if not (np.isfinite(self.sampling_rate) and self.sampling_rate > 0.0):
            raise ValueError(f"sampling_rate must be positive and finite, got {self.sampling_rate}")

    @property
    def nyquist_frequency(self) -> float:
        """
        Get the highest frequency the sampling rate can represent.

        Returns
        -------
        float
            Half of ``sampling_rate``, in hertz.
        """
        return 0.5 * self.sampling_rate

    @property
    @override
    def minimum_length(self) -> int:
        """
        Get the fewest samples a signal needs to be filtered.

        Returns
        -------
        int
            One for ``Phase.CAUSAL``. For ``Phase.ZERO``, one more than the
            samples the signal is extended by at each end to settle the filter.
        """
        match self.phase:
            case Phase.ZERO:
                return self._padding_length + 1
            case Phase.CAUSAL:
                return 1

    def frequency_response(self, frequencies: ArrayLike | None = None) -> FrequencyResponse:
        """
        Compute the response of :meth:`apply` to sinusoids.

        With ``Phase.ZERO`` the response is ``|H(f)|^2``: real, non-negative, and
        without phase shift.

        Parameters
        ----------
        frequencies : ArrayLike | None
            The frequencies in hertz, a 1-D array within ``[0, nyquist_frequency]``.
            None means 512 evenly spaced frequencies from 0 to the Nyquist
            frequency.

        Returns
        -------
        FrequencyResponse
            The complex gain at each frequency.

        Raises
        ------
        TypeError
            If the frequencies are not real numbers.
        ValueError
            If the frequencies are not 1-D, not finite, or outside
            ``[0, nyquist_frequency]``.
        """
        evaluated = self._as_frequencies(frequencies)
        response = self._transfer(evaluated)
        match self.phase:
            case Phase.ZERO:
                response = (np.abs(response) ** 2).astype(np.complex128)
            case Phase.CAUSAL:
                pass
        return FrequencyResponse(frequencies=evaluated, response=response)

    @override
    def _filter(self, samples: FloatArray) -> FloatArray:
        match self.phase:
            case Phase.ZERO:
                return self._forward_backward(samples)
            case Phase.CAUSAL:
                return self._forward(samples)

    def _require_below_nyquist(self, frequency: float, name: str) -> None:
        """
        Check that a frequency lies strictly between 0 and the Nyquist frequency.

        Raises
        ------
        ValueError
            If it does not.
        """
        if not 0.0 < frequency < self.nyquist_frequency:
            raise ValueError(
                f"{name} must lie strictly between 0 and the Nyquist frequency "
                + f"{self.nyquist_frequency} Hz, got {frequency}"
            )

    def _require_band(self, band: Band) -> None:
        """
        Check that every edge of a band lies strictly between 0 and the Nyquist frequency.

        Raises
        ------
        ValueError
            If one does not.
        """
        for edge in band.edges:
            self._require_below_nyquist(edge, "band edge")

    @staticmethod
    def _band_edges(band: Band) -> float | tuple[float, ...]:
        """Pass a single edge as a scalar, as the SciPy design functions expect."""
        return band.edges[0] if len(band.edges) == 1 else band.edges

    @staticmethod
    def _band_name(band_type: BandType) -> Literal["lowpass", "highpass", "bandpass", "bandstop"]:
        """Name a band type as the SciPy design functions do."""
        match band_type:
            case BandType.LOWPASS:
                return "lowpass"
            case BandType.HIGHPASS:
                return "highpass"
            case BandType.BANDPASS:
                return "bandpass"
            case BandType.BANDSTOP:
                return "bandstop"

    def _as_frequencies(self, frequencies: ArrayLike | None) -> FloatArray:
        if frequencies is None:
            return np.linspace(0.0, self.nyquist_frequency, _DEFAULT_RESPONSE_POINTS)
        array = np.asarray(frequencies)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"frequencies must be real numbers, got dtype {array.dtype}")
        evaluated: FloatArray = np.array(array, dtype=np.float64)
        if evaluated.ndim != 1:
            raise ValueError(f"frequencies must be 1-D, got shape {evaluated.shape}")
        if not np.isfinite(evaluated).all():
            raise ValueError("frequencies must be finite")
        if evaluated.size > 0 and (
            evaluated.min() < 0.0 or evaluated.max() > self.nyquist_frequency
        ):
            raise ValueError(
                f"frequencies must lie within [0, {self.nyquist_frequency}] Hz, "
                + f"got [{evaluated.min()}, {evaluated.max()}]"
            )
        return evaluated

    @property
    @abstractmethod
    def _padding_length(self) -> int:
        """
        Get the samples a ``Phase.ZERO`` run extends the signal by at each end.

        Returns
        -------
        int
            The padding length.
        """

    @abstractmethod
    def _forward(self, samples: FloatArray) -> FloatArray:
        """
        Run the filter forward once, starting in the steady state of the first sample.

        Parameters
        ----------
        samples : FloatArray
            Finite samples with shape ``(n_samples, *channel_shape)``.

        Returns
        -------
        FloatArray
            The filtered samples with the same shape.
        """

    @abstractmethod
    def _forward_backward(self, samples: FloatArray) -> FloatArray:
        """
        Run the filter forward and backward over the signal padded by odd extension.

        Parameters
        ----------
        samples : FloatArray
            Finite samples with shape ``(n_samples, *channel_shape)``, longer than
            ``_padding_length``.

        Returns
        -------
        FloatArray
            The filtered samples with the same shape.
        """

    @abstractmethod
    def _transfer(self, frequencies: FloatArray) -> ComplexArray:
        """
        Evaluate the transfer function of one forward pass.

        Parameters
        ----------
        frequencies : FloatArray
            Frequencies in hertz within ``[0, nyquist_frequency]``.

        Returns
        -------
        ComplexArray
            The complex gain at each frequency.
        """
