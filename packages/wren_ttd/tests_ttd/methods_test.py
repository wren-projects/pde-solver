# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

import warnings

import numpy as np
import pytest
from wren_common.tests import arange_tensor
from wren_ttd import TTD
from wren_ttd._numpy_api import DenseWarning

from .common import assert_default_epsilon


def test_reduction_methods_delegate() -> None:
    """Test that the reduction methods agree with NumPy."""
    dense = arange_tensor(3, 4)
    ttd = TTD(dense)

    assert_default_epsilon(ttd.sum(axis=0), np.sum(dense, axis=0))
    assert_default_epsilon(ttd.mean(axis=0), np.mean(dense, axis=0))
    assert_default_epsilon(ttd.cumsum(axis=1), np.cumsum(dense, axis=1))


def test_trace_method_delegates() -> None:
    """Test that the trace method agrees with NumPy."""
    dense = arange_tensor(3, 3)
    ttd = TTD(dense)

    assert_default_epsilon(ttd.trace(), np.trace(dense))


def test_reshape_and_flatten_methods_delegate() -> None:
    """Test the reshape, ravel, and flatten methods."""
    dense = arange_tensor(2, 3)
    ttd = TTD(dense)

    assert_default_epsilon(ttd.reshape((3, 2)), np.reshape(dense, (3, 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DenseWarning)
        assert_default_epsilon(ttd.ravel(), np.ravel(dense))
        assert_default_epsilon(ttd.flatten(), np.ravel(dense))


def test_selection_methods_delegate() -> None:
    """Test the take, repeat, and tile methods."""
    dense = arange_tensor(2, 3)
    ttd = TTD(dense)

    assert_default_epsilon(ttd.take([1, 0], axis=0), np.take(dense, [1, 0], axis=0))
    assert_default_epsilon(ttd.repeat(2, axis=0), np.repeat(dense, 2, axis=0))
    assert_default_epsilon(ttd.tile((2, 1)), np.tile(dense, (2, 1)))


def test_sum_method_accepts_numpy_signature() -> None:
    """Test dtype, out, and keepdims through the method."""
    dense = arange_tensor(2, 3)
    ttd = TTD(dense)

    scalar = ttd.sum(dtype=np.float32)
    assert scalar.dtype == np.dtype(np.float32)

    out = TTD.zeros((2, 1), dtype=np.float64)
    result = ttd.sum(axis=1, keepdims=True, out=out)
    assert result is out
    assert_default_epsilon(out, np.sum(dense, axis=1, keepdims=True))

    with pytest.raises(NotImplementedError, match="initial"):
        _ = ttd.sum(initial=1.0)
