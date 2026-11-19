"""NumPy array type aliases used across the package."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]
type BoolArray = NDArray[np.bool_]
