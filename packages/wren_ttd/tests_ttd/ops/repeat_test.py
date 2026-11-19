# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize("axis", [0, -1])
def test_repeat_vector(axis: int) -> None:
    """Test repeating every element of a vector."""
    dense = arange_tensor(4)
    ttd = TTD(dense)
    expected = np.repeat(dense, 2, axis=axis)

    assert_default_epsilon(ops.repeat(ttd, 2, axis=axis), expected)
    assert_default_epsilon(np.repeat(ttd, 2, axis=axis), expected)


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_repeat_matrix(axis: int) -> None:
    """Test repeating every element of a matrix along each axis."""
    dense = arange_tensor(3, 4)
    ttd = TTD(dense)
    expected = np.repeat(dense, 2, axis=axis)

    assert_default_epsilon(ops.repeat(ttd, 2, axis=axis), expected)
    assert_default_epsilon(np.repeat(ttd, 2, axis=axis), expected)


@pytest.mark.parametrize(
    ("repeats", "axis"),
    [([0, 2, 1], 0), ([2, 0, 1, 3], 1), ([2, 0, 1, 3], -1)],
)
def test_repeat_per_index(repeats: Sequence[int], axis: int) -> None:
    """Test per-index repetitions, including zero repetitions."""
    dense = arange_tensor(3, 4)
    ttd = TTD(dense)
    expected = np.repeat(dense, repeats, axis=axis)

    assert_default_epsilon(ops.repeat(ttd, repeats, axis=axis), expected)
    assert_default_epsilon(np.repeat(ttd, repeats, axis=axis), expected)


@pytest.mark.parametrize(
    ("repeats", "axis"),
    [(2, 1), ([1, 2, 0], 1), (2, -2), ([2, 0, 1], -2)],
)
def test_repeat_3d(repeats: int | Sequence[int], axis: int) -> None:
    """Test repeating elements of a 3-D tensor."""
    dense = arange_tensor(2, 3, 4)
    ttd = TTD(dense)
    expected = np.repeat(dense, repeats, axis=axis)

    assert_default_epsilon(ops.repeat(ttd, repeats, axis=axis), expected)
    assert_default_epsilon(np.repeat(ttd, repeats, axis=axis), expected)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_repeat_flattened(shape: tuple[int, ...]) -> None:
    """Test repeating elements of the flattened tensor."""
    dense = arange_tensor(*shape)
    ttd = TTD(dense)
    expected = np.repeat(dense, 2)

    assert_default_epsilon(ops.repeat(ttd, 2), expected)
    assert_default_epsilon(np.repeat(ttd, 2), expected)

    repeats = np.arange(dense.size) % 3
    expected_per_index = np.repeat(dense, repeats)
    assert_default_epsilon(ops.repeat(ttd, repeats), expected_per_index)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("axis", [0, -1])
def test_repeat_test_tensors(tensor: TestTensor, ttd: TestTTD, axis: int) -> None:
    """Test repeating elements of the shared test tensors."""
    expected = np.repeat(tensor, 2, axis=axis)

    assert_default_epsilon(ops.repeat(ttd, 2, axis=axis), expected)
    assert_default_epsilon(np.repeat(ttd, 2, axis=axis), expected)


@pytest.mark.parametrize("repeats", [-1, [1, -1, 2]])
def test_repeat_negative_rejected(repeats: int | Sequence[int]) -> None:
    """Test that negative repetitions are rejected."""
    ttd = TTD(arange_tensor(3))

    with pytest.raises(ValueError, match="negative"):
        _ = ops.repeat(ttd, repeats, axis=0)
    with pytest.raises(ValueError, match="negative"):
        _ = np.repeat(ttd, repeats, axis=0)
