# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestCumsum:
    """Tests for TTD cumulative sums."""

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_axis(self, tensor: TestTensor, ttd: TestTTD, axis: int) -> None:
        """Test cumulative sums along an axis via both entry points."""
        expected = np.cumsum(tensor, axis=axis)
        assert_default_epsilon(
            ops.cumulative_sum(ttd, axis=axis), expected, scale=tensor
        )
        assert_default_epsilon(np.cumsum(ttd, axis=axis), expected, scale=tensor)

    def test_axis_none(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test the documented dense fallback for axis=None."""
        expected = np.cumsum(tensor, axis=None)
        assert_default_epsilon(
            ops.cumulative_sum(ttd), expected, scale=tensor, epsilon=1e-8
        )
        assert_default_epsilon(np.cumsum(ttd), expected, scale=tensor, epsilon=1e-8)

    def test_preserves_shape_and_ranks(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that an integer-axis cumsum preserves shape and TT ranks."""
        result = ops.cumulative_sum(ttd, axis=0)
        assert ttd.shape == tensor.shape
        assert result.shape == ttd.shape
        assert result.ranks == ttd.ranks

    def test_dtype(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that dtype sets the result dtype."""
        assert ttd.shape == tensor.shape
        result = ops.cumulative_sum(ttd, axis=0, dtype=np.float32)
        assert result.dtype == np.dtype(np.float32)

        dense_result = np.cumsum(ttd, dtype=np.float32)
        assert dense_result.dtype == np.dtype(np.float32)

    def test_dtype_values(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test float32 cumsum values."""
        expected = np.cumsum(tensor, axis=1, dtype=np.float32)
        result = ops.cumulative_sum(ttd, axis=1, dtype=np.float32)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)

    def test_out_ttd(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test writing the result into an explicit output."""
        expected = np.cumsum(tensor, axis=0)
        out = TTD.zeros(ttd.shape, dtype=ttd.dtype)

        result = ops.cumulative_sum(ttd, axis=0, out=out)
        assert result is out
        assert_default_epsilon(out, expected, scale=tensor)

        numpy_out = TTD.zeros(ttd.shape, dtype=ttd.dtype)
        numpy_result = cast(Any, np.cumsum)(ttd, axis=0, out=numpy_out)
        assert numpy_result is numpy_out
        assert_default_epsilon(numpy_out, expected, scale=tensor)


def test_cumsum_out_dtype() -> None:
    """Test that out can narrow a float64 result to float32."""
    tensor = arange_tensor(2, 3)
    ttd = TTD(tensor)
    expected = np.cumsum(tensor, axis=0)
    out = TTD.zeros(ttd.shape, dtype=np.float32)

    result = ops.cumulative_sum(ttd, axis=0, out=out)
    assert result is out
    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-5)


def test_cumsum_invalid_axis() -> None:
    """Test that out-of-range axes are rejected."""
    ttd = TTD(arange_tensor(2, 3))
    with pytest.raises(np.exceptions.AxisError):
        _ = ops.cumulative_sum(ttd, axis=2)


def test_cumsum_non_floating_dtype() -> None:
    """Test that only floating-point result dtypes are accepted."""
    ttd = TTD(arange_tensor(2, 3))
    with pytest.raises(TypeError, match="dtype"):
        _ = ops.cumulative_sum(ttd, dtype=np.int64)


def test_cumsum_axis_none_is_1d() -> None:
    """Test that the flattened fallback returns a 1-D TTD."""
    tensor = arange_tensor(2, 3)
    result = ops.cumulative_sum(TTD(tensor))
    assert isinstance(result, TTD)
    assert result.shape == (tensor.size,)
    np.testing.assert_allclose(np.asarray(result), np.cumsum(tensor))
