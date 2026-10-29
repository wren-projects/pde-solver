"""Benchmarks for operations that require a dense array."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from wren_common.types import NDArray
from wren_ttd import TTD

from .common import DENSE_CASES, TensorCase, random_ttd


@pytest.fixture(params=DENSE_CASES, ids=[case.id for case in DENSE_CASES])
def dense_ttd(request: pytest.FixtureRequest) -> TTD[np.float64]:
    """
    Return a random TTD for the current dense benchmark case.

    The dense shapes are restricted by ``DENSE_MAX_TENSOR_BYTES`` so that the
    expanded tensor and the temporaries of compression fit in roughly 8 GiB.

    """
    case: TensorCase = request.param
    return random_ttd(case.shape, case.rank)


@pytest.fixture
def dense_array(dense_ttd: TTD[np.float64]) -> NDArray[np.float64]:
    """Expand the current TTD into a dense array for compression benchmarks."""
    return np.asarray(dense_ttd)


def test_compression(
    dense_array: NDArray[np.float64], benchmark: BenchmarkFixture
) -> None:
    """Benchmark compression of a dense array into a TTD."""
    benchmark(TTD.from_ndarray, dense_array)


def test_expansion(dense_ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark expansion of a TTD into a dense array."""
    benchmark(np.asarray, dense_ttd)
