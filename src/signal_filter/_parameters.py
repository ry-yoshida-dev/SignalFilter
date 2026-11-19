"""Private validation of filter parameters that type hints cannot enforce at runtime."""

from __future__ import annotations

import numbers


def require_integer(value: object, name: str) -> None:
    """
    Check that a parameter is an integer.

    Booleans are rejected even though Python treats them as integers.

    Raises
    ------
    TypeError
        If ``value`` is not an integer, such as a float or a boolean.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__} {value!r}")
