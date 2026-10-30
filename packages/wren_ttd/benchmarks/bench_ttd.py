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


def test_add(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark element-wise addition of two TTDs (TTD + TTD)."""
    benchmark(ops.add, ttd, ttd)


def test_subtract_scalar(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark subtraction of a scalar."""
    benchmark(ops.subtract, ttd, 1.0)


def test_subtract(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark element-wise subtraction of two TTDs (TTD - TTD)."""
    benchmark(ops.subtract, ttd, ttd)


def test_multiply_scalar(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark multiplication by a scalar."""
    benchmark(ops.multiply, ttd, 2.0)


def test_multiply(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark the Hadamard (element-wise) product of two TTDs."""
    benchmark(ops.multiply, ttd, ttd)


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


def test_tensordot_axes0(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark tensordot with axes=0 (concatenation, no contraction)."""
    benchmark(ops.tensordot, ttd, ttd, axes=0)


def test_tensordot_axes2(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark contraction of the last two modes with the first two."""
    benchmark(ops.tensordot, ttd, ttd, axes=2)


def test_tensordot_all_axes(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark full contraction of two TTDs into a scalar."""
    benchmark(ops.tensordot, ttd, ttd, axes=ttd.ndim)


def test_tensordot_first_axis(
    ttd: TTD[np.float64], benchmark: BenchmarkFixture
) -> None:
    """Benchmark contraction along the first mode of both TTDs."""
    benchmark(ops.tensordot, ttd, ttd, axes=(0, 0))


def test_tensordot_last_axis(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark contraction along the last mode of both TTDs."""
    benchmark(ops.tensordot, ttd, ttd, axes=(-1, -1))


def test_round(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark one full rounding sweep."""

    def setup() -> tuple[tuple[TTD[np.float64]], dict[str, object]]:
        return (ttd.copy(),), {}

    benchmark.pedantic(
        lambda x: x.round(epsilon=0.0),
        setup=setup,
        rounds=3,
    )


def test_neg(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark element-wise negation."""
    benchmark(ops.neg, ttd)


def test_transpose(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark reversing the order of all axes."""
    benchmark(ops.transpose, ttd)


def test_transpose_axes(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark a non-trivial axis permutation."""
    axes = (1, 0, *range(2, ttd.ndim))
    benchmark(ops.transpose, ttd, axes)


def test_swapaxes(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark swapping the first and last axes."""
    benchmark(ops.swapaxes, ttd, 0, -1)


def test_setitem_zero(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark setting a single entry to zero."""

    def setup() -> tuple[tuple[TTD[np.float64]], dict[str, object]]:
        return (ttd.copy(),), {}

    benchmark.pedantic(
        lambda x: x.__setitem__((0,), 0.0),
        setup=setup,
    )


def test_setitem_zero_slice(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark zeroing a slice of the first mode."""

    def setup() -> tuple[tuple[TTD[np.float64]], dict[str, object]]:
        return (ttd.copy(),), {}

    benchmark.pedantic(
        lambda x: x.__setitem__((slice(1, -1),), 0.0),
        setup=setup,
    )


def test_setitem_nonzero(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark setting a single entry to a nonzero value."""

    def setup() -> tuple[tuple[TTD[np.float64]], dict[str, object]]:
        return (ttd.copy(),), {}

    benchmark.pedantic(
        lambda x: x.__setitem__((0,), 1.0),
        setup=setup,
    )


def test_setitem_nonzero_slice(
    ttd: TTD[np.float64], benchmark: BenchmarkFixture
) -> None:
    """Benchmark setting a slice of the first mode to a nonzero value."""

    def setup() -> tuple[tuple[TTD[np.float64]], dict[str, object]]:
        return (ttd.copy(),), {}

    benchmark.pedantic(
        lambda x: x.__setitem__((slice(1, -1),), 1.0),
        setup=setup,
    )


def test_zeros_like(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark creating a zero-filled TTD with the same shape."""
    benchmark(ops.zeros_like, ttd)


def test_ones_like(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark creating a one-filled TTD with the same shape."""
    benchmark(ops.ones_like, ttd)


def test_full_like(ttd: TTD[np.float64], benchmark: BenchmarkFixture) -> None:
    """Benchmark creating a constant-filled TTD with the same shape."""
    benchmark(ops.full_like, ttd, 2.0)
