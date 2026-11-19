# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


def test_tile_vector() -> None:
    """Test tiling a vector with a scalar repetition count."""
    dense = arange_tensor(4)
    ttd = TTD(dense)
    expected = np.tile(dense, 3)

    assert_default_epsilon(ops.tile(ttd, 3), expected)
    assert_default_epsilon(np.tile(ttd, 3), expected)


@pytest.mark.parametrize(
    "reps",
    [2, (2,), [2], (2, 3), (1, 2)],
)
def test_tile_matrix(reps: int | Sequence[int]) -> None:
    """Test tiling a matrix with scalar and per-axis repetitions."""
    dense = arange_tensor(3, 4)
    ttd = TTD(dense)
    expected = np.tile(dense, reps)

    assert_default_epsilon(ops.tile(ttd, reps), expected)
    assert_default_epsilon(np.tile(ttd, reps), expected)


def test_tile_shorter_tuple() -> None:
    """Test that a short reps tuple is prepended with ones."""
    dense = arange_tensor(3, 4)
    ttd = TTD(dense)

    assert_default_epsilon(ops.tile(ttd, (1,)), dense)
    assert ops.tile(ttd, (1,)).shape == dense.shape


@pytest.mark.parametrize(
    ("reps", "expected_shape"),
    [((2, 1, 2), (2, 3, 8)), ((1, 2, 3), (1, 6, 12)), ((2, 2), (6, 8))],
)
def test_tile_longer_tuple(
    reps: Sequence[int], expected_shape: tuple[int, ...]
) -> None:
    """Test that extra reps entries add new leading axes."""
    dense = arange_tensor(3, 4)
    ttd = TTD(dense)
    expected = np.tile(dense, reps)

    result = ops.tile(ttd, reps)
    assert result.shape == expected_shape
    assert_default_epsilon(result, expected)
    assert_default_epsilon(np.tile(ttd, reps), expected)


@pytest.mark.parametrize("reps", [(2, 1, 3), (1, 2, 1), 2, (2,)])
def test_tile_3d(reps: int | Sequence[int]) -> None:
    """Test tiling a 3-D tensor."""
    dense = arange_tensor(2, 3, 4)
    ttd = TTD(dense)
    expected = np.tile(dense, reps)

    assert_default_epsilon(ops.tile(ttd, reps), expected)
    assert_default_epsilon(np.tile(ttd, reps), expected)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_tile_test_tensors(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test tiling the shared test tensors."""
    expected = np.tile(tensor, (2,))

    assert_default_epsilon(ops.tile(ttd, (2,)), expected)
    assert_default_epsilon(np.tile(ttd, (2,)), expected)


@pytest.mark.parametrize("reps", [-1, (2, -1)])
def test_tile_negative_rejected(reps: int | Sequence[int]) -> None:
    """Test that negative repetitions are rejected."""
    ttd = TTD(arange_tensor(2, 3))

    with pytest.raises(ValueError, match="negative"):
        _ = ops.tile(ttd, reps)
    with pytest.raises(ValueError, match="negative"):
        _ = np.tile(ttd, reps)
