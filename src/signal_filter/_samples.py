"""Private conversion of user signals into validated sample arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .array_types import FloatArray


def as_samples(signal: ArrayLike, minimum_length: int) -> FloatArray:
    """
    Convert ``signal`` into a float64 copy with samples along the first axis.

    Raises
    ------
    TypeError
        If the signal is not real-valued numbers (booleans, complex numbers, and
        objects are rejected).
    ValueError
        If the signal is 0-D, has fewer than ``minimum_length`` samples, or
        contains NaN or infinity.
    """
    array = np.asarray(signal)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"signal must hold real integers or floats, got dtype {array.dtype}")
    if array.ndim == 0:
        raise ValueError("signal must have at least one dimension, with samples along the first")
    if array.shape[0] < minimum_length:
        raise ValueError(
            f"signal must have at least {minimum_length} samples, got {array.shape[0]}"
        )
    samples: FloatArray = np.array(array, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError("signal must contain only finite values; fill gaps before filtering")
    return samples
