# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from wren_common.tests import arange_tensor
from wren_ttd import TTD, ops

from ..common import EpsilonComparable, TestTTD, assert_default_epsilon


class TestTrace:
    """Tests for TTD traces."""

    def test_2d_default(self) -> None:
        """Test the default axes of a 2-D tensor."""
        tensor = arange_tensor(4, 4)
        ttd = TTD(tensor)
        expected = np.trace(tensor)

        for result in (ops.trace(ttd), np.trace(ttd)):
            assert np.isscalar(result)
            assert_default_epsilon(
                cast(EpsilonComparable, result), expected, scale=tensor
            )

    def test_2d_swapped_axes(self) -> None:
        """Test that swapping the two axes gives the same trace."""
        tensor = arange_tensor(4, 4)
        ttd = TTD(tensor)
        expected = np.trace(tensor)

        assert_default_epsilon(ops.trace(ttd, axis1=1, axis2=0), expected, scale=tensor)
        assert_default_epsilon(np.trace(ttd, axis1=1, axis2=0), expected, scale=tensor)

    def test_3d_default(self) -> None:
        """Test the default axes of a 3-D tensor."""
        tensor = arange_tensor(3, 3, 2)
        ttd = TTD(tensor)
        expected = np.trace(tensor)

        result = ops.trace(ttd)
        assert isinstance(result, TTD)
        assert result.shape == expected.shape
        assert_default_epsilon(result, expected, scale=tensor)

        numpy_result = np.trace(ttd)
        assert isinstance(numpy_result, TTD)
        assert numpy_result.shape == expected.shape
        assert_default_epsilon(cast(TestTTD, numpy_result), expected, scale=tensor)

    def test_3d_swapped_axes(self) -> None:
        """Test swapped axes of a 3-D tensor."""
        tensor = arange_tensor(3, 3, 2)
        ttd = TTD(tensor)
        expected = np.trace(tensor, axis1=1, axis2=0)

        result = ops.trace(ttd, axis1=1, axis2=0)
        assert isinstance(result, TTD)
        assert result.shape == expected.shape
        assert_default_epsilon(result, expected, scale=tensor)

        assert_default_epsilon(np.trace(ttd, axis1=1, axis2=0), expected, scale=tensor)

    def test_non_final_axes(self) -> None:
        """Test tracing axes that have to be moved to the end."""
        tensor = arange_tensor(2, 3, 4, 3)
        ttd = TTD(tensor)
        expected = np.trace(tensor, axis1=1, axis2=3)

        result = ops.trace(ttd, axis1=1, axis2=3)
        assert isinstance(result, TTD)
        assert result.shape == expected.shape
        assert_default_epsilon(result, expected, scale=tensor)

        assert_default_epsilon(np.trace(ttd, axis1=1, axis2=3), expected, scale=tensor)

    def test_offset_rejected(self) -> None:
        """Test that nonzero offsets are rejected."""
        ttd = TTD(arange_tensor(3, 3))
        with pytest.raises(NotImplementedError, match="offset"):
            _ = ops.trace(ttd, offset=1)
        with pytest.raises(NotImplementedError, match="offset"):
            _ = np.trace(ttd, offset=1)

    def test_same_axis_rejected(self) -> None:
        """Test that the two axes must differ."""
        ttd = TTD(arange_tensor(3, 3))
        with pytest.raises(ValueError, match="axis1 and axis2"):
            _ = ops.trace(ttd, axis1=0, axis2=0)

    def test_unequal_axes_rejected(self) -> None:
        """Test that the two traced axes must have the same size."""
        ttd = TTD(arange_tensor(2, 3))
        with pytest.raises(ValueError, match="same size"):
            _ = ops.trace(ttd)

    def test_dtype(self) -> None:
        """Test the scalar and TTD result dtypes."""
        tensor = arange_tensor(3, 3)
        scalar = ops.trace(TTD(tensor), dtype=np.float32)
        assert scalar.dtype == np.dtype(np.float32)

        result = cast(TestTTD, np.trace(TTD(arange_tensor(3, 3, 2)), dtype=np.float32))
        assert isinstance(result, TTD)
        assert result.dtype == np.dtype(np.float32)

    def test_out_ttd(self) -> None:
        """Test writing a TTD result into an explicit output."""
        tensor = arange_tensor(3, 3, 2)
        ttd = TTD(tensor)
        expected = np.trace(tensor)
        out = TTD.zeros(expected.shape, dtype=ttd.dtype)

        result = ops.trace(ttd, out=out)
        assert result is out
        assert_default_epsilon(out, expected, scale=tensor)

        numpy_out = TTD.zeros(expected.shape, dtype=ttd.dtype)
        numpy_result = cast(Any, np.trace)(ttd, out=numpy_out)
        assert numpy_result is numpy_out
        assert_default_epsilon(numpy_out, expected, scale=tensor)

    def test_out_scalar_rejected(self) -> None:
        """Test that a scalar trace cannot be written into a TTD."""
        ttd = TTD(arange_tensor(3, 3))
        out = TTD.zeros((1,), dtype=ttd.dtype)
        with pytest.raises(NotImplementedError, match="out"):
            _ = ops.trace(ttd, out=out)
