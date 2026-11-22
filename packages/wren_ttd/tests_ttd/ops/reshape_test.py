# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import ops
from wren_ttd.core import TTD

from ..common import TEST_TTD, TestTTD, assert_default_epsilon

VECTOR = arange_tensor(6)
MATRIX = arange_tensor(2, 3)
TENSOR_3D = arange_tensor(2, 3, 4)


@pytest.mark.parametrize(
    ("source_shape", "target_shape"),
    [
        ((2, 3), (6,)),
        ((2, 3, 4), (2, 12)),
        ((2, 3, 4), (6, 4)),
        ((6,), (2, 3)),
        ((12,), (2, 3, 2)),
        ((2, 6), (3, 4)),
        ((5, 4), (2, 10)),
        ((2, 3, 4), (24,)),
        ((2, 3, 4), (2, 2, 6)),
        ((2, 3, 4), (3, 2, 4)),
        ((3, 5), (1, 3, 5)),
    ],
)
def test_reshape(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> None:
    """Test merging, splitting, and interleaved reshapes against NumPy."""
    dense = arange_tensor(*source_shape)
    ttd = TTD(dense)
    expected = dense.reshape(target_shape)

    assert_default_epsilon(ops.reshape(ttd, target_shape), expected)
    assert_default_epsilon(np.reshape(ttd, target_shape), expected)


@pytest.mark.parametrize("shape", [6, (6,), [2, 3]])
def test_reshape_shape_forms(shape: int | tuple[int, ...] | list[int]) -> None:
    """Test int, tuple, and list shape arguments."""
    ttd = TTD(MATRIX)
    expected = MATRIX.reshape(shape)
    assert_default_epsilon(ops.reshape(ttd, shape), expected)
    assert_default_epsilon(np.reshape(ttd, shape), expected)


@pytest.mark.parametrize("target_shape", [(-1, 2), (2, -1), (-1,), (4, -1, 2)])
def test_reshape_negative_one(target_shape: tuple[int, ...]) -> None:
    """Test inference of a single unknown dimension."""
    dense = arange_tensor(2, 3, 4)
    ttd = TTD(dense)
    expected = dense.reshape(target_shape)

    assert_default_epsilon(ops.reshape(ttd, target_shape), expected)
    assert_default_epsilon(np.reshape(ttd, target_shape), expected)


@pytest.mark.parametrize("order", ["F", "A", "K"])
def test_reshape_rejects_order(order: str) -> None:
    """Test that non-C orders are rejected."""
    ttd = TTD(MATRIX)
    with pytest.raises(NotImplementedError):
        _ = ops.reshape(ttd, (6,), order=order)
    with pytest.raises(NotImplementedError):
        _ = np.reshape(ttd, (6,), order=cast(Any, order))


@pytest.mark.parametrize(
    ("target_shape", "match"),
    [
        ((-1, -1), "can only specify one unknown dimension"),
        ((2, -2), "negative dimensions"),
        ((4,), "cannot reshape array of size"),
        ((2, 2), "cannot reshape array of size"),
    ],
)
def test_reshape_rejects_invalid_shapes(
    target_shape: tuple[int, ...], match: str
) -> None:
    """Test that invalid shapes raise a ValueError."""
    ttd = TTD(VECTOR)
    with pytest.raises(ValueError, match=match):
        _ = ops.reshape(ttd, target_shape)
    with pytest.raises(ValueError, match=match):
        _ = np.reshape(ttd, target_shape)


def test_reshape_copy_is_noop() -> None:
    """Test that the NumPy ``copy`` keyword is accepted and ignored."""
    ttd = TTD(MATRIX)
    expected = MATRIX.reshape(6)

    assert_default_epsilon(np.reshape(ttd, (6,), copy=None), expected)
    assert_default_epsilon(np.reshape(ttd, (6,), copy=True), expected)
    assert_default_epsilon(np.reshape(ttd, (6,), copy=False), expected)


def test_reshape_same_shape() -> None:
    """Test that reshaping to the same shape keeps the values."""
    ttd = TTD(TENSOR_3D)
    assert_default_epsilon(ops.reshape(ttd, ttd.shape), TENSOR_3D)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reshape_preserves_dtype(dtype: type[np.floating]) -> None:
    """Test that the result keeps the input dtype."""
    ttd = TTD(TENSOR_3D.astype(dtype))
    assert ops.reshape(ttd, (6, 4)).dtype == dtype
    assert np.reshape(ttd, (6, 4)).dtype == dtype


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_reshape_round_trip(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that reshaping back to the original shape restores the tensor."""
    intermediate = ops.reshape(ttd, (tensor.size,))
    restored = ops.reshape(intermediate, tensor.shape)
    assert_default_epsilon(restored, tensor)
