"""Checks for the Wu et al. 2023 matrix conversion."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wu_matrix.datasets import FEATURE_COLUMNS, list_batteries, read_battery_csv
from wu_matrix.transform import dominant_periods, reshape_period, series_to_matrices


def test_sine_period_is_8():
    cycles = np.arange(64)
    signal = np.sin(2 * np.pi * cycles / 8)
    series = np.repeat(signal[:, None], 3, axis=1)
    frequencies, periods, amplitudes = dominant_periods(series, k=1)
    assert int(frequencies[0]) == 8
    assert int(periods[0]) == 8
    assert amplitudes.shape == (1,)


def test_each_row_is_one_period():
    series = np.arange(8, dtype=np.float64).reshape(8, 1)
    matrix = reshape_period(series, period=4)
    assert matrix.shape == (2, 4, 1)
    assert matrix[:, :, 0].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_tail_is_zero_padded():
    series = np.arange(5, dtype=np.float64).reshape(5, 1)
    matrix = reshape_period(series, period=4)
    assert matrix.shape == (2, 4, 1)
    assert matrix[1, :, 0].tolist() == [4, 0, 0, 0]


def test_original_values_stay_in_cycle_order():
    series = np.arange(20, dtype=np.float64).reshape(10, 2)
    converted = series_to_matrices(series, k=2)
    first = converted["matrices"][0]
    period = int(converted["periods"][0])
    flat = first.reshape(-1, 2)[:10]
    assert np.allclose(flat, series)
    assert first.shape[1] == period
    assert first.shape[2] == 2


def test_mit_and_hust_tables_convert():
    for dataset, expected in [("MIT", 125), ("HUST", 77)]:
        batteries = list_batteries(dataset)
        assert len(batteries) == expected
        features, capacity = read_battery_csv(batteries[0]["path"])
        assert features.shape[1] == len(FEATURE_COLUMNS)
        assert len(capacity) == len(features)
        converted = series_to_matrices(features, k=5)
        assert len(converted["matrices"]) == 5
        for matrix, period in zip(converted["matrices"], converted["periods"]):
            assert matrix.shape[1] == int(period)
            assert matrix.shape[2] == 16
            assert matrix.shape[0] * matrix.shape[1] >= len(features)
            assert np.isfinite(matrix).all()
            # The first cycles occupy the start of the matrix, before padding.
            head = matrix.reshape(-1, 16)[: len(features)]
            assert np.allclose(head, features)
