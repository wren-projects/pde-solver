# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import TTD, ops

from ..common import (
    TEST_TTD,
    EpsilonComparable,
    TestTTD,
    assert_default_epsilon,
)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestSum:
    """Tests for TTD summation."""

    def test_full(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test summation over all axes via all entry points."""
        expected = np.sum(tensor)
        for result in (ttd.sum(), ops.sum(ttd), np.sum(ttd)):
            assert np.isscalar(result)
            assert_default_epsilon(cast(EpsilonComparable, result), expected)

    @pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1), ()])
    def test_axis(
        self,
        tensor: TestTensor,
        ttd: TestTTD,
        axis: int | tuple[int, ...] | None,
    ) -> None:
        """Test summation over a single axis, a tuple, None, and ()."""
        expected = np.sum(tensor, axis=axis)
        assert_default_epsilon(ops.sum(ttd, axis), expected)
        assert_default_epsilon(ttd.sum(axis), expected)
        assert_default_epsilon(np.sum(ttd, axis=axis), expected)

    @pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
    def test_keepdims(
        self,
        tensor: TestTensor,
        ttd: TestTTD,
        axis: int | tuple[int, ...] | None,
    ) -> None:
        """Test that keepdims retains reduced axes with size one."""
        expected = np.sum(tensor, axis=axis, keepdims=True)

        result = ops.sum(ttd, axis=axis, keepdims=True)
        assert isinstance(result, TTD)
        assert result.shape == expected.shape
        assert_default_epsilon(result, expected)

        numpy_result = np.sum(ttd, axis=axis, keepdims=True)
        assert isinstance(numpy_result, TTD)
        assert numpy_result.shape == expected.shape
        assert_default_epsilon(cast(TestTTD, numpy_result), expected)

    def test_axis_empty_is_copy(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that reducing no axes returns an independent copy."""
        result = ops.sum(ttd, axis=())
        assert isinstance(result, TTD)
        assert result.shape == ttd.shape
        assert_default_epsilon(result, tensor)
        result.data[0][...] = 0
        assert not np.all(np.asarray(ttd) == 0)

    def test_dtype(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that dtype sets the scalar and TTD result dtypes."""
        assert ttd.shape == tensor.shape
        scalar = ops.sum(ttd, dtype=np.float32)
        assert scalar.dtype == np.dtype(np.float32)

        result = np.sum(ttd, axis=0, dtype=np.float32)
        assert isinstance(result, TTD)
        assert result.dtype == np.dtype(np.float32)

    def test_dtype_values(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test float32 summation values."""
        expected = np.sum(tensor, axis=0, dtype=np.float32)
        result = ops.sum(ttd, axis=0, dtype=np.float32)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4)

    def test_out_ttd(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test writing a TTD result into an explicit output."""
        expected = np.sum(tensor, axis=0)
        out = TTD.zeros(expected.shape, dtype=ttd.dtype)

        result = ops.sum(ttd, axis=0, out=out)
        assert result is out
        assert_default_epsilon(out, expected)

        numpy_out = TTD.zeros(expected.shape, dtype=ttd.dtype)
        numpy_result = cast(Any, np.sum)(ttd, axis=0, out=numpy_out)
        assert numpy_result is numpy_out
        assert_default_epsilon(numpy_out, expected)

    def test_out_scalar_rejected(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that scalar results cannot be written into a TTD."""
        assert ttd.shape == tensor.shape
        out = TTD.zeros((1,), dtype=ttd.dtype)
        with pytest.raises(NotImplementedError, match="out"):
            _ = ops.sum(ttd, out=out)
        with pytest.raises(NotImplementedError, match="out"):
            _ = cast(Any, np.sum)(ttd, out=out)

    def test_initial_rejected(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that the initial value is rejected explicitly."""
        assert ttd.shape == tensor.shape
        with pytest.raises(NotImplementedError, match="initial"):
            _ = ops.sum(ttd, initial=1.0)
        with pytest.raises(NotImplementedError, match="initial"):
            _ = np.sum(ttd, initial=1.0)

    def test_where_rejected(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that only the default where=True is accepted."""
        mask = np.ones(tensor.shape, dtype=bool)
        with pytest.raises(NotImplementedError, match="where"):
            _ = ops.sum(ttd, where=mask)
        with pytest.raises(NotImplementedError, match="where"):
            _ = np.sum(ttd, where=False)

    def test_where_true(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that where=True is a no-op."""
        assert_default_epsilon(ops.sum(ttd, where=True), np.sum(tensor))
        assert_default_epsilon(np.sum(ttd, where=True), np.sum(tensor))

    def test_dtype_from_cast(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test a TTD result cast to a different dtype."""
        result = ops.sum(ttd, axis=0, dtype=np.float64)
        assert result.dtype == np.dtype(np.float64)
        assert_default_epsilon(result, np.sum(tensor, axis=0))


def test_sum_invalid_axis() -> None:
    """Test that out-of-range axes are rejected."""
    ttd = TTD(arange_tensor(2, 3))
    with pytest.raises(np.exceptions.AxisError):
        _ = ops.sum(ttd, axis=2)


def test_sum_duplicate_axis() -> None:
    """Test that duplicate axes are rejected."""
    ttd = TTD(arange_tensor(2, 3))
    with pytest.raises(ValueError, match="repeated"):
        _ = ops.sum(ttd, axis=(0, 0))


def test_sum_non_floating_dtype() -> None:
    """Test that only floating-point result dtypes are accepted."""
    ttd = TTD(arange_tensor(2, 3))
    with pytest.raises(TypeError, match="dtype"):
        _ = ops.sum(ttd, dtype=np.int64)


def test_sum_scalar_result_dtype() -> None:
    """Test that the scalar result uses the TTD dtype."""
    ttd = TTD(arange_tensor(2, 3), dtype=np.float32)
    result = cast(np.float32, ops.sum(ttd))
    assert result.dtype == np.dtype(np.float32)
