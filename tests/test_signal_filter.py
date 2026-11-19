"""Behavior every filter shares: shapes, channels, dtypes, and input validation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from signal_filter import FloatArray, SignalFilter


def _noisy_signal(n_samples: int, *channel_shape: int) -> FloatArray:
    generator = np.random.default_rng(0)
    return generator.standard_normal((n_samples, *channel_shape)) + 5.0


def test_output_keeps_shape_and_is_float64(any_filter: SignalFilter) -> None:
    signal = _noisy_signal(500)
    filtered = any_filter.apply(signal)
    assert filtered.shape == signal.shape
    assert filtered.dtype == np.float64
    assert np.isfinite(filtered).all()


def test_channels_are_filtered_independently(any_filter: SignalFilter) -> None:
    signal = _noisy_signal(500, 2, 3)
    filtered = any_filter.apply(signal)
    assert filtered.shape == signal.shape
    for row in range(2):
        for column in range(3):
            np.testing.assert_allclose(
                filtered[:, row, column], any_filter.apply(signal[:, row, column]), atol=1e-12
            )


def test_input_is_not_modified(any_filter: SignalFilter) -> None:
    signal = _noisy_signal(500)
    signal[100] = 50.0
    original = signal.copy()
    any_filter.apply(signal)
    np.testing.assert_array_equal(signal, original)


def test_integers_and_lists_are_accepted(any_filter: SignalFilter) -> None:
    integers = np.arange(500) % 7
    np.testing.assert_allclose(
        any_filter.apply(integers), any_filter.apply(integers.astype(np.float64))
    )
    np.testing.assert_allclose(any_filter.apply(integers.tolist()), any_filter.apply(integers))


def test_signal_of_minimum_length_is_filtered(any_filter: SignalFilter) -> None:
    filtered = any_filter.apply(np.ones(any_filter.minimum_length))
    assert filtered.shape == (any_filter.minimum_length,)


def test_shorter_signal_is_rejected(any_filter: SignalFilter) -> None:
    with pytest.raises(ValueError, match="at least"):
        any_filter.apply(np.ones(any_filter.minimum_length - 1))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(any_filter: SignalFilter, value: float) -> None:
    signal = np.ones(500)
    signal[10] = value
    with pytest.raises(ValueError, match="finite"):
        any_filter.apply(signal)


@pytest.mark.parametrize(
    "signal",
    [np.ones(500, dtype=np.bool_), np.ones(500, dtype=np.complex128), np.array(["a"] * 500)],
    ids=["bool", "complex", "string"],
)
def test_non_real_signals_are_rejected(
    any_filter: SignalFilter, signal: NDArray[np.generic]
) -> None:
    with pytest.raises(TypeError, match="real"):
        any_filter.apply(signal)


def test_scalar_signal_is_rejected(any_filter: SignalFilter) -> None:
    with pytest.raises(ValueError, match="dimension"):
        any_filter.apply(1.0)
