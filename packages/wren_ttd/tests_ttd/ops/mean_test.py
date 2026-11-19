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
class TestMean:
    """Tests for TTD arithmetic means."""

    @pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1), ()])
    def test_axis(
        self,
        tensor: TestTensor,
        ttd: TestTTD,
        axis: int | tuple[int, ...] | None,
    ) -> None:
        """Test the mean over single axes, tuples, None, and ()."""
        expected = np.mean(tensor, axis=axis)
        assert_default_epsilon(ops.mean(ttd, axis), expected)
        assert_default_epsilon(np.mean(ttd, axis=axis), expected)

    @pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
    def test_keepdims(
        self,
        tensor: TestTensor,
        ttd: TestTTD,
        axis: int | tuple[int, ...] | None,
    ) -> None:
        """Test that keepdims retains reduced axes with size one."""
        expected = np.mean(tensor, axis=axis, keepdims=True)

        result = ops.mean(ttd, axis=axis, keepdims=True)
        assert isinstance(result, TTD)
        assert result.shape == expected.shape
        assert_default_epsilon(result, expected)

        numpy_result = np.mean(ttd, axis=axis, keepdims=True)
        assert isinstance(numpy_result, TTD)
        assert numpy_result.shape == expected.shape
        assert_default_epsilon(cast(TestTTD, numpy_result), expected)

    def test_axis_empty_is_copy(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that reducing no axes returns the input values."""
        result = ops.mean(ttd, axis=())
        assert isinstance(result, TTD)
        assert result.shape == ttd.shape
        assert_default_epsilon(result, tensor)

    def test_dtype(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that float32 means stay float32."""
        assert ttd.shape == tensor.shape
        scalar = ops.mean(ttd, dtype=np.float32)
        assert scalar.dtype == np.dtype(np.float32)

        result = np.mean(ttd, axis=0, dtype=np.float32)
        assert isinstance(result, TTD)
        assert result.dtype == np.dtype(np.float32)

    def test_dtype_values(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test float32 mean values."""
        expected = np.mean(tensor, axis=0, dtype=np.float32)
        result = ops.mean(ttd, axis=0, dtype=np.float32)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4)

    def test_out_ttd(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test writing a TTD result into an explicit output."""
        expected = np.mean(tensor, axis=0)
        out = TTD.zeros(expected.shape, dtype=ttd.dtype)

        result = ops.mean(ttd, axis=0, out=out)
        assert result is out
        assert_default_epsilon(out, expected)

        numpy_out = TTD.zeros(expected.shape, dtype=ttd.dtype)
        numpy_result = cast(Any, np.mean)(ttd, axis=0, out=numpy_out)
        assert numpy_result is numpy_out
        assert_default_epsilon(numpy_out, expected)

    def test_out_scalar_rejected(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that scalar results cannot be written into a TTD."""
        assert ttd.shape == tensor.shape
        out = TTD.zeros((1,), dtype=ttd.dtype)
        with pytest.raises(NotImplementedError, match="out"):
            _ = ops.mean(ttd, out=out)
        with pytest.raises(NotImplementedError, match="out"):
            _ = cast(Any, np.mean)(ttd, out=out)

    def test_where_rejected(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that only the default where=True is accepted."""
        mask = np.ones(tensor.shape, dtype=bool)
        with pytest.raises(NotImplementedError, match="where"):
            _ = ops.mean(ttd, where=mask)
        with pytest.raises(NotImplementedError, match="where"):
            _ = np.mean(ttd, where=False)

    def test_where_true(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test that where=True is a no-op."""
        assert_default_epsilon(ops.mean(ttd, where=True), np.mean(tensor))
        assert_default_epsilon(np.mean(ttd, where=True), np.mean(tensor))


def test_mean_invalid_axis() -> None:
    """Test that out-of-range axes are rejected."""
    ttd = TTD(arange_tensor(2, 3))
    with pytest.raises(np.exceptions.AxisError):
        _ = ops.mean(ttd, axis=2)


def test_mean_non_floating_dtype() -> None:
    """Test that only floating-point result dtypes are accepted."""
    ttd = TTD(arange_tensor(2, 3))
    with pytest.raises(TypeError, match="dtype"):
        _ = ops.mean(ttd, dtype=np.int64)
