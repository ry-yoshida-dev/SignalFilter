"""Tests for the moving average, median, Savitzky-Golay, and Hampel filters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import pytest
from scipy import signal as scipy_signal

import signal_filter.sliding_window_filters.hampel as hampel_module
from signal_filter import (
    Boundary,
    FloatArray,
    HampelFilter,
    MedianFilter,
    MovingAverageFilter,
    SavitzkyGolayFilter,
)

_PADDING_MODES: dict[Boundary, Literal["reflect", "edge", "wrap"]] = {
    Boundary.REFLECT: "reflect",
    Boundary.NEAREST: "edge",
    Boundary.WRAP: "wrap",
}


def _windows(samples: FloatArray, window_length: int, boundary: Boundary) -> FloatArray:
    """Every centered window, padded independently of the library with ``np.pad``."""
    half = window_length // 2
    padded = np.pad(samples, (half, half), mode=_PADDING_MODES[boundary])
    return np.lib.stride_tricks.sliding_window_view(padded, window_length)


def _random_walk(n_samples: int) -> FloatArray:
    return np.cumsum(np.random.default_rng(1).standard_normal(n_samples))


@pytest.mark.parametrize("boundary", Boundary)
@pytest.mark.parametrize(("n_samples", "window_length"), [(50, 1), (50, 7), (5, 9), (2, 15)])
def test_moving_average_and_median_follow_the_boundary(
    boundary: Boundary, n_samples: int, window_length: int
) -> None:
    samples = _random_walk(n_samples)
    windows = _windows(samples, window_length, boundary)
    moving_average = MovingAverageFilter(window_length=window_length, boundary=boundary)
    median = MedianFilter(window_length=window_length, boundary=boundary)
    np.testing.assert_allclose(moving_average.apply(samples), windows.mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(median.apply(samples), np.median(windows, axis=1))


@pytest.mark.parametrize("boundary", Boundary)
def test_savitzky_golay_follows_the_boundary(boundary: Boundary) -> None:
    samples = _random_walk(50)
    windows = _windows(samples, 9, boundary)
    coefficients = scipy_signal.savgol_coeffs(9, 3, use="dot")
    savitzky_golay = SavitzkyGolayFilter(window_length=9, polynomial_order=3, boundary=boundary)
    np.testing.assert_allclose(savitzky_golay.apply(samples), windows @ coefficients, atol=1e-12)


@pytest.mark.parametrize("boundary", Boundary)
def test_hampel_follows_the_boundary(boundary: Boundary) -> None:
    samples = _random_walk(50)
    samples[[0, 20, 49]] += 30.0
    windows = _windows(samples, 7, boundary)
    medians = np.median(windows, axis=1)
    deviations = np.median(np.abs(windows - medians[:, np.newaxis]), axis=1)
    is_outlier = np.abs(samples - medians) > 3.0 * 1.4826 * deviations
    hampel = HampelFilter(window_length=7, boundary=boundary)
    np.testing.assert_array_equal(hampel.detect(samples), is_outlier)
    np.testing.assert_array_equal(hampel.apply(samples), np.where(is_outlier, medians, samples))


def test_moving_average_keeps_constants_and_lines_away_from_the_ends() -> None:
    line = 2.0 * np.arange(30) + 1.0
    moving_average = MovingAverageFilter(window_length=5)
    np.testing.assert_allclose(moving_average.apply(np.full(30, 3.0)), 3.0)
    np.testing.assert_allclose(moving_average.apply(line)[2:-2], line[2:-2])


def test_median_removes_impulses_and_keeps_steps() -> None:
    step = np.repeat([0.0, 1.0], 20)
    spiky = step.copy()
    spiky[[5, 6, 30]] = 9.0
    np.testing.assert_array_equal(MedianFilter(window_length=5).apply(spiky), step)


def test_savitzky_golay_reproduces_polynomials_of_its_order() -> None:
    position = np.linspace(-1.0, 1.0, 40)
    cubic = 2.0 * position**3 - position + 0.5
    savitzky_golay = SavitzkyGolayFilter(window_length=9, polynomial_order=3)
    np.testing.assert_allclose(savitzky_golay.apply(cubic)[4:-4], cubic[4:-4], atol=1e-12)


def test_savitzky_golay_differentiates_with_the_sample_spacing() -> None:
    spacing = 0.01
    time = np.arange(100) * spacing
    quadratic = 3.0 * time**2 - time
    slope = SavitzkyGolayFilter(
        window_length=7, polynomial_order=2, derivative=1, sample_spacing=spacing
    )
    np.testing.assert_allclose(slope.apply(quadratic)[3:-3], (6.0 * time - 1.0)[3:-3], atol=1e-9)


@pytest.mark.parametrize("polynomial_order", [0, 1])
def test_low_order_savitzky_golay_is_a_moving_average(polynomial_order: int) -> None:
    samples = _random_walk(40)
    savitzky_golay = SavitzkyGolayFilter(window_length=7, polynomial_order=polynomial_order)
    np.testing.assert_allclose(
        savitzky_golay.apply(samples), MovingAverageFilter(window_length=7).apply(samples)
    )


def test_hampel_replaces_only_the_outliers() -> None:
    clean = np.sin(np.linspace(0.0, 6.0, 200)) + 0.01 * np.random.default_rng(2).standard_normal(
        200
    )
    corrupted = clean.copy()
    spikes = [10, 57, 58, 150]
    corrupted[spikes] += [5.0, -4.0, -4.0, 3.0]
    hampel = HampelFilter(window_length=11)
    is_outlier = hampel.detect(corrupted)
    assert is_outlier[spikes].all()
    # Gaussian noise exceeds three robust standard deviations about 0.3 % of the time.
    assert np.count_nonzero(is_outlier) - len(spikes) <= 2
    repaired = hampel.apply(corrupted)
    np.testing.assert_array_equal(repaired[~is_outlier], corrupted[~is_outlier])
    np.testing.assert_allclose(repaired[spikes], clean[spikes], atol=0.05)


def test_hampel_with_zero_threshold_is_a_median_filter() -> None:
    samples = _random_walk(60)
    np.testing.assert_allclose(
        HampelFilter(window_length=5, threshold=0.0).apply(samples),
        MedianFilter(window_length=5).apply(samples),
    )


def test_hampel_detects_per_channel() -> None:
    samples = np.zeros((30, 2))
    samples[12, 1] = 5.0
    is_outlier = HampelFilter(window_length=5).detect(samples)
    assert is_outlier.shape == (30, 2)
    np.testing.assert_array_equal(np.argwhere(is_outlier), [[12, 1]])


def test_hampel_in_chunks_matches_one_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = _random_walk(103)
    samples[[0, 40, 102]] += 20.0
    hampel = HampelFilter(window_length=9)
    expected_outliers = hampel.detect(samples)
    expected = hampel.apply(samples)
    # Room for two windows of the random walk per chunk, with a remainder at the end.
    monkeypatch.setattr(hampel_module, "_CHUNK_ELEMENTS", 2 * 9)
    np.testing.assert_array_equal(hampel.detect(samples), expected_outliers)
    np.testing.assert_array_equal(hampel.apply(samples), expected)


def test_hampel_detection_validates_the_signal() -> None:
    with pytest.raises(ValueError, match="finite"):
        HampelFilter(window_length=5).detect([1.0, np.nan])


type Build = Callable[[], object]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: MovingAverageFilter(window_length=4), "odd"),
        (lambda: MedianFilter(window_length=0), "positive"),
        (lambda: SavitzkyGolayFilter(window_length=5, polynomial_order=5), "polynomial_order"),
        (lambda: SavitzkyGolayFilter(window_length=5, polynomial_order=-1), "polynomial_order"),
        (lambda: SavitzkyGolayFilter(window_length=5, derivative=3), "derivative"),
        (lambda: SavitzkyGolayFilter(window_length=5, sample_spacing=0.0), "sample_spacing"),
        (lambda: HampelFilter(window_length=5, threshold=-1.0), "threshold"),
        (lambda: HampelFilter(window_length=5, threshold=np.inf), "threshold"),
    ],
)
def test_invalid_parameters_are_rejected(build: Build, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize(
    ("build", "name"),
    [
        (lambda: MovingAverageFilter(window_length=cast(int, 5.0)), "window_length"),
        (lambda: MedianFilter(window_length=cast(int, True)), "window_length"),
        (
            lambda: SavitzkyGolayFilter(window_length=5, polynomial_order=cast(int, 2.0)),
            "polynomial_order",
        ),
        (lambda: SavitzkyGolayFilter(window_length=5, derivative=cast(int, 1.0)), "derivative"),
    ],
)
def test_non_integer_parameters_are_rejected(build: Build, name: str) -> None:
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        build()
