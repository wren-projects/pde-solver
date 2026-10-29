"""Benchmarks for operations that stay in the compressed TTD representation."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from wren_ttd import TTD, ops

from .common import TT_CASES, TensorCase, random_ttd


@pytest.fixture(params=TT_CASES, ids=[case.id for case in TT_CASES])
def ttd(request: pytest.FixtureRequest) -> TTD[np.float64]:
    """
    Return a fresh random TTD for the current benchmark case.

    The tensor is generated directly from its cores, not by compressing a
    dense array, so even the largest cases stay small in memory.

    """
    case: TensorCase = request.param
    return random_ttd(case.shape, case.rank)


def test_copy(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark copying the compressed representation."""
    benchmark(ttd.copy)


def test_get_item_scalar(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark indexing all modes with integers."""
    index = (0,) * ttd.ndim
    benchmark(ttd.__getitem__, index)


def test_get_item_slice(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark slicing the first mode."""
    benchmark(ttd.__getitem__, (0,))


def test_add_scalar(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark addition of a scalar, which increases every rank by one."""
    benchmark(ops.add, ttd, 1.0)


def test_multiply_scalar(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark multiplication by a scalar."""
    benchmark(ops.multiply, ttd, 2.0)


def test_frobenius_norm(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark the Frobenius norm."""
    benchmark(ops.frobenius_norm, ttd)


def test_inner_product(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark the inner product of a TTD with itself."""
    benchmark(ops.inner_product, ttd, ttd)


def test_gradient(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark differentiation along a single mode."""
    benchmark(ops.gradient, ttd, axis=0)


def test_pad(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark constant padding by one element on each side."""
    benchmark(ops.pad, ttd, 1, constant_values=0.0)


def test_stack(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark stacking two copies along a new leading mode."""
    benchmark(ops.stack, [ttd, ttd])


def test_tensordot(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark contraction of the last mode with the first mode."""
    benchmark(ops.tensordot, ttd, ttd, axes=1)


def test_round(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """
    Benchmark one full rounding sweep.

    The truncation tolerance is set to zero so that the random, full-rank
    cores are not collapsed to rank one and the cost of the QR/SVD sweeps is
    measured instead of just the cost of a single truncation.

    """
    benchmark.pedantic(ttd.round, kwargs={"epsilon": 0.0}, rounds=3)
