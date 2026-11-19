# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from wren_common.tests import TestTensorPair, arange_tensor
from wren_ttd import TTD, ops

from ..common import TEST_PAIR_TTD, TestTTDPair, assert_default_epsilon


def test_concat_1d() -> None:
    """Test concatenating 1-D TTDs."""
    dense = arange_tensor(3)
    other = arange_tensor(4) + 10
    a, b = TTD(dense), TTD(other)
    expected = np.concatenate([dense, other])

    assert_default_epsilon(ops.concat([a, b]), expected)
    assert_default_epsilon(np.concatenate([a, b]), expected)
    assert_default_epsilon(np.concat([a, b], axis=0), expected)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((4, 4), 0),
        ((2, 4), 0),
        ((3, 2), 1),
        ((3, 2), -1),
        ((2, 4), -2),
    ],
)
def test_concat_2d(shape: tuple[int, ...], axis: int) -> None:
    """Test concatenating matrices along each axis."""
    dense = arange_tensor(3, 4)
    other = arange_tensor(*shape) + 100
    a, b = TTD(dense), TTD(other)
    expected = np.concatenate([dense, other], axis=axis)

    assert_default_epsilon(ops.concat([a, b], axis=axis), expected)
    assert_default_epsilon(np.concatenate([a, b], axis=axis), expected)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((1, 3, 4), 0),
        ((2, 2, 4), 1),
        ((2, 3, 1), 2),
        ((2, 2, 4), -2),
    ],
)
def test_concat_3d(shape: tuple[int, ...], axis: int) -> None:
    """Test concatenating 3-D TTDs."""
    dense = arange_tensor(2, 3, 4)
    other = arange_tensor(*shape) + 100
    a, b = TTD(dense), TTD(other)
    expected = np.concatenate([dense, other], axis=axis)

    assert_default_epsilon(ops.concat([a, b], axis=axis), expected)
    assert_default_epsilon(np.concatenate([a, b], axis=axis), expected)


def test_concat_three_inputs() -> None:
    """Test concatenating more than two TTDs."""
    dense = arange_tensor(2, 3, 4)
    others = [arange_tensor(2, 1, 4) + offset for offset in (100, 200)]
    ttds = [TTD(dense), *(TTD(other) for other in others)]
    expected = np.concatenate([dense, *others], axis=1)

    assert_default_epsilon(ops.concat(ttds, axis=1), expected)
    assert_default_epsilon(np.concatenate(ttds, axis=1), expected)


def test_concat_flattened() -> None:
    """Test concatenating the flattened inputs."""
    a_dense, b_dense = arange_tensor(3, 4), arange_tensor(2, 4) + 100
    a, b = TTD(a_dense), TTD(b_dense)
    expected = np.concatenate([a_dense.ravel(), b_dense.ravel()])

    assert_default_epsilon(ops.concat([a, b], axis=None), expected)
    assert_default_epsilon(np.concatenate([a, b], axis=None), expected)


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
@pytest.mark.parametrize("axis", [0, -1])
def test_concat_test_tensors(
    tensors: TestTensorPair, ttds: TestTTDPair, axis: int
) -> None:
    """Test concatenating the shared test tensor pairs."""
    expected = np.concatenate(tensors, axis=axis)

    assert_default_epsilon(ops.concat(ttds, axis=axis), expected)
    assert_default_epsilon(np.concatenate(ttds, axis=axis), expected)


def test_concat_dtype_promotion() -> None:
    """Test that mixed float dtypes promote to float64."""
    a: TTD[Any] = TTD.full((2, 3), 1.0, dtype=np.float32)
    b: TTD[Any] = TTD.full((2, 3), 2.0, dtype=np.float64)

    result = ops.concat([a, b], axis=0)
    assert result.dtype == np.dtype(np.float64)
    assert_default_epsilon(result, np.concatenate([np.asarray(a), np.asarray(b)]))

    numpy_result = np.concatenate([a, b], axis=0)
    assert numpy_result.dtype == np.dtype(np.float64)


def test_concat_dtype_override() -> None:
    """Test that dtype= overrides the promoted result dtype."""
    a: TTD[Any] = TTD.full((2, 3), 1.0, dtype=np.float32)
    b: TTD[Any] = TTD.full((2, 3), 2.0, dtype=np.float64)

    result = ops.concat([a, b], axis=0, dtype=np.float32)
    assert result.dtype == np.dtype(np.float32)
    assert_default_epsilon(result, np.concatenate([np.asarray(a), np.asarray(b)]))

    numpy_result = np.concatenate([a, b], axis=0, dtype=np.float32)
    assert numpy_result.dtype == np.dtype(np.float32)


def test_concat_shape_mismatch() -> None:
    """Test that mismatched non-concatenated axes are rejected."""
    a = TTD(arange_tensor(3, 4))
    b = TTD(arange_tensor(2, 4))

    with pytest.raises(ValueError, match="Shape mismatch"):
        _ = ops.concat([a, b], axis=1)
    with pytest.raises(ValueError, match="Shape mismatch"):
        _ = np.concatenate([a, b], axis=1)


def test_concat_ndim_mismatch() -> None:
    """Test that inputs with different dimensions are rejected."""
    a = TTD(arange_tensor(4))
    b = TTD(arange_tensor(2, 2))

    with pytest.raises(ValueError, match="same number of dimensions"):
        _ = ops.concat([a, b])


def test_concat_empty() -> None:
    """Test that an empty input sequence is rejected."""
    with pytest.raises(ValueError, match="at least one"):
        _ = ops.concat([])


def test_concat_out_rejected() -> None:
    """Test that out= is rejected."""
    a = TTD(arange_tensor(2, 3))
    out: TTD[Any] = TTD.zeros((4, 3))

    with pytest.raises(NotImplementedError):
        _ = ops.concat([a, a], out=out)


def test_concat_casting_rejected() -> None:
    """Test that unsupported casting modes are rejected."""
    a = TTD(arange_tensor(2, 3))

    with pytest.raises(NotImplementedError):
        _ = ops.concat([a, a], casting="unsafe")
    with pytest.raises(NotImplementedError):
        _ = np.concatenate([a, a], casting="unsafe")
