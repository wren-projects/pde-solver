# pyright: reportAny=false, reportUnknownVariableType=false
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from wren_common.tests import EDGE_TEST_SCALARS, SMALL_TEST_SCALARS, TestTensor
from wren_ttd import TTD

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_zeros_like(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test zeros_like."""
    result = np.zeros_like(ttd)
    assert isinstance(result, TTD)
    assert_default_epsilon(result, np.zeros_like(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_ones_like(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test ones_like."""
    result = np.ones_like(ttd)
    assert isinstance(result, TTD)
    assert_default_epsilon(result, np.ones_like(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("fill_value", deepcopy(SMALL_TEST_SCALARS))
def test_full_like(tensor: TestTensor, ttd: TestTTD, fill_value: float) -> None:
    """Test full_like."""
    result = np.full_like(ttd, fill_value)
    assert isinstance(result, TTD)
    assert_default_epsilon(result, np.full_like(tensor, fill_value))

    dtype = np.dtype(np.float32)
    shape = (2, 2)

    result = np.full_like(ttd, 2.5, dtype=dtype, shape=shape)
    assert_default_epsilon(result, np.full_like(tensor, 2.5, dtype=dtype, shape=shape))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD[:2]))
@pytest.mark.parametrize("fill_value", EDGE_TEST_SCALARS)
def test_full_like_edge(tensor: TestTensor, ttd: TestTTD, fill_value: float) -> None:
    """Test full_like with large-magnitude edge values."""
    result = np.full_like(ttd, fill_value)
    assert isinstance(result, TTD)
    assert_default_epsilon(result, np.full_like(tensor, fill_value), scale=fill_value)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("func", [np.zeros_like, np.ones_like])
@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.dtype(np.float32), "float32", float],
    ids=["type", "dtype", "str", "builtin"],
)
def test_like_dtype_and_shape(
    func: Callable[..., np.ndarray], tensor: TestTensor, ttd: TestTTD, dtype: Any
) -> None:
    """Test that the like functions accept a dtype and shape override."""
    result = func(ttd, dtype=dtype)
    assert isinstance(result, TTD)
    assert_default_epsilon(result, func(tensor, dtype=dtype))

    shape = (2, 2)
    result = func(ttd, shape=shape)
    assert isinstance(result, TTD)
    assert_default_epsilon(result, func(tensor, shape=shape))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD[:1]))
def test_like_numpy_arguments(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that the like functions handle the remaining NumPy arguments."""
    dense = np.ones_like(ttd, subok=False)
    assert isinstance(dense, np.ndarray)
    assert_default_epsilon(dense, np.ones_like(tensor, subok=False))

    result = np.zeros_like(ttd, order="F")
    assert isinstance(result, TTD)
    assert_default_epsilon(result, np.zeros_like(tensor, order="F"))
