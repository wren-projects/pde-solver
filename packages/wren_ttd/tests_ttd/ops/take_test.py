# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


def test_take_scalar_vector() -> None:
    """Test taking scalar indices from a vector."""
    dense = arange_tensor(5)
    ttd = TTD(dense)

    for index in (0, 3, -1, -5):
        expected = np.take(dense, index)
        assert isinstance(ops.take(ttd, index), np.floating)
        assert_default_epsilon(ops.take(ttd, index), expected)
        assert_default_epsilon(np.take(ttd, index), expected)


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
@pytest.mark.parametrize("index", [1, -2])
def test_take_scalar_matrix(axis: int, index: int) -> None:
    """Test taking a scalar index from a matrix drops the selected axis."""
    dense = arange_tensor(3, 4)
    ttd = TTD(dense)
    expected = np.take(dense, index, axis=axis)

    assert_default_epsilon(ops.take(ttd, index, axis=axis), expected)
    assert_default_epsilon(np.take(ttd, index, axis=axis), expected)


@pytest.mark.parametrize(
    ("indices", "axis"),
    [
        ([2, 0], 0),
        ((3, 1, 0), 1),
        (np.array([-1, 0]), 1),
        ([0, 2], -2),
    ],
)
def test_take_matrix_indices(
    indices: Sequence[int] | npt.NDArray[np.intp], axis: int
) -> None:
    """Test taking integer arrays from a matrix on every axis."""
    dense = arange_tensor(3, 4)
    ttd = TTD(dense)
    expected = np.take(dense, indices, axis=axis)

    assert_default_epsilon(ops.take(ttd, indices, axis=axis), expected)
    assert_default_epsilon(np.take(ttd, indices, axis=axis), expected)


@pytest.mark.parametrize(
    ("indices", "axis"),
    [([1, 0], 0), ([2, 0, 1], 1), ([3, 1, 0], 2), ([2, 0, 1], -2)],
)
def test_take_3d(indices: Sequence[int], axis: int) -> None:
    """Test taking indices from a 3-D tensor."""
    dense = arange_tensor(2, 3, 4)
    ttd = TTD(dense)
    expected = np.take(dense, indices, axis=axis)

    assert_default_epsilon(ops.take(ttd, indices, axis=axis), expected)
    assert_default_epsilon(np.take(ttd, indices, axis=axis), expected)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 2)])
def test_take_flattened(shape: tuple[int, ...]) -> None:
    """Test taking indices from the flattened tensor."""
    dense = arange_tensor(*shape)
    ttd = TTD(dense)
    indices = [0, -1, 2]
    expected = np.take(dense, indices, axis=None)

    assert_default_epsilon(ops.take(ttd, indices), expected)
    assert_default_epsilon(np.take(ttd, indices), expected)
    assert_default_epsilon(ops.take(ttd, 3), np.take(dense, 3, axis=None))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("axis", [0, -1])
def test_take_test_tensors(tensor: TestTensor, ttd: TestTTD, axis: int) -> None:
    """Test taking indices from the shared test tensors."""
    indices = [0, -1]
    expected = np.take(tensor, indices, axis=axis)

    assert_default_epsilon(ops.take(ttd, indices, axis=axis), expected)
    assert_default_epsilon(np.take(ttd, indices, axis=axis), expected)


@pytest.mark.parametrize("mode", ["wrap", "clip"])
def test_take_mode_rejected(mode: Literal["wrap", "clip"]) -> None:
    """Test that unsupported take modes are rejected."""
    ttd = TTD(arange_tensor(3, 4))

    with pytest.raises(NotImplementedError):
        _ = ops.take(ttd, [0], axis=0, mode=mode)
    with pytest.raises(NotImplementedError):
        _ = np.take(ttd, [0], axis=0, mode=mode)


@pytest.mark.parametrize("indices", [4, -5, [0, 4], [0, -5]])
def test_take_out_of_bounds(indices: int | Sequence[int]) -> None:
    """Test that out-of-range indices are rejected."""
    ttd = TTD(arange_tensor(4))

    with pytest.raises(IndexError):
        _ = ops.take(ttd, indices, axis=0)
    with pytest.raises(IndexError):
        _ = np.take(ttd, indices, axis=0)


def test_take_out() -> None:
    """Test writing a take result into an explicit output TTD."""
    dense = arange_tensor(3, 4)
    ttd = TTD(dense)
    indices = [2, 0]
    expected = np.take(dense, indices, axis=0)

    out: TTD[Any] = TTD.zeros((2, 4))
    assert ops.take(ttd, indices, axis=0, out=out) is out
    assert_default_epsilon(out, expected)

    numpy_out: TTD[Any] = TTD.zeros((2, 4))
    result = cast(Any, np.take)(ttd, indices, axis=0, out=numpy_out)
    assert result is numpy_out
    assert_default_epsilon(numpy_out, expected)
