"""Complex frequency response of a filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .array_types import ComplexArray, FloatArray


# Arrays compare element-wise, so instances compare and hash by identity instead of by value.
@dataclass(frozen=True, eq=False)
class FrequencyResponse:
    """
    Response of a filter to a sinusoid at each frequency.

    Attributes
    ----------
    frequencies : FloatArray
        The frequencies in hertz, with shape ``(n_frequencies,)``.
    response : ComplexArray
        The complex gain at each frequency.
    """

    frequencies: FloatArray
    response: ComplexArray

    @property
    def magnitude(self) -> FloatArray:
        """
        Get the amplitude gain.

        Returns
        -------
        FloatArray
            ``|H(f)|``; 1 passes a sinusoid unchanged and 0 removes it.
        """
        return np.abs(self.response)

    @property
    def gain_db(self) -> FloatArray:
        """
        Get the amplitude gain in decibels.

        Returns
        -------
        FloatArray
            ``20 log10 |H(f)|``; ``-inf`` where the response is exactly zero.
        """
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(self.magnitude)

    @property
    def phase(self) -> FloatArray:
        """
        Get the phase shift.

        Returns
        -------
        FloatArray
            The unwrapped phase angle in radians.
        """
        return np.unwrap(np.angle(self.response))
