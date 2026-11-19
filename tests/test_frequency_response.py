"""Tests for frequency responses."""

from __future__ import annotations

import numpy as np

from signal_filter import FrequencyResponse


def test_magnitude_gain_and_phase() -> None:
    response = FrequencyResponse(
        frequencies=np.array([0.0, 1.0, 2.0, 3.0]),
        response=np.array([1.0, 0.1j, -0.01, 0.0], dtype=np.complex128),
    )
    np.testing.assert_allclose(response.magnitude, [1.0, 0.1, 0.01, 0.0])
    np.testing.assert_allclose(response.gain_db[:3], [0.0, -20.0, -40.0])
    assert response.gain_db[3] == -np.inf
    np.testing.assert_allclose(response.phase[:3], [0.0, np.pi / 2, np.pi])


def test_responses_compare_and_hash_by_identity() -> None:
    def build() -> FrequencyResponse:
        return FrequencyResponse(
            frequencies=np.array([0.0, 1.0]), response=np.array([1.0, 0.5], dtype=np.complex128)
        )

    response = build()
    assert response == response
    assert response != build()
    assert len({response, response, build()}) == 2
