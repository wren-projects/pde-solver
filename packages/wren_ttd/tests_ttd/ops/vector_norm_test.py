# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((4,), None),
        ((4,), 0),
        ((4,), -1),
        ((3, 4), None),
        ((3, 4), 0),
        ((3, 4), 1),
        ((3, 4), -1),
        ((3, 4), (0, 1)),
        ((2, 3, 4), None),
        ((2, 3, 4), 1),
        ((2, 3, 4), (0, 2)),
        ((2, 3, 4), -2),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_vector_norm(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
    keepdims: bool,  # noqa: FBT001
) -> None:
    """Compare vector norms for vectors, matrices, and 3-D tensors."""
    tensor = arange_tensor(*shape)
    ttd = TTD(tensor)
    numpy_axis: Any = axis
    expected = np.linalg.vector_norm(tensor, axis=numpy_axis, keepdims=keepdims)

    assert_default_epsilon(
        np.linalg.vector_norm(ttd, axis=numpy_axis, keepdims=keepdims),
        expected,
        scale=tensor,
    )
    assert_default_epsilon(
        ops.vector_norm(ttd, axis=axis, keepdims=keepdims),
        expected,
        scale=tensor,
    )


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_vector_norm_full_reduction(tensor: TestTensor, ttd: TestTTD) -> None:
    """Compare the full vector norm against NumPy."""
    assert_default_epsilon(np.linalg.vector_norm(ttd), np.linalg.vector_norm(tensor))
    assert_default_epsilon(ops.vector_norm(ttd), np.linalg.vector_norm(tensor))
    assert isinstance(np.linalg.vector_norm(ttd), np.floating)
    assert isinstance(ops.vector_norm(ttd), np.floating)


@pytest.mark.parametrize("ord", [None, 2, 2.0])
def test_vector_norm_accepts_euclidean_order(ord: float | None) -> None:  # noqa: A002
    """Accept the forms of the Euclidean order used by NumPy."""
    tensor = arange_tensor(3, 4)
    ttd = TTD(tensor)
    expected = np.linalg.vector_norm(tensor, ord=ord)

    assert_default_epsilon(np.linalg.vector_norm(ttd, ord=ord), expected, scale=tensor)
    assert_default_epsilon(ops.vector_norm(ttd, ord=ord), expected, scale=tensor)


@pytest.mark.parametrize("ord", [1, -1, np.inf, "fro"])
def test_vector_norm_rejects_other_orders(ord: float | str) -> None:  # noqa: A002
    """Reject norms other than the Euclidean norm."""
    ttd = TTD.ones((3, 4))
    numpy_ord: Any = ord

    with pytest.raises(NotImplementedError, match="Euclidean"):
        _ = np.linalg.vector_norm(ttd, ord=numpy_ord)

    with pytest.raises(NotImplementedError, match="Euclidean"):
        _ = ops.vector_norm(ttd, ord=ord)


def test_vector_norm_keepdims_shape() -> None:
    """Keep every reduced axis as a size-one dimension."""
    ttd = TTD(arange_tensor(2, 3, 4))

    result = ops.vector_norm(ttd, axis=(0, 2), keepdims=True)

    assert isinstance(result, TTD)
    assert result.shape == (1, 3, 1)


def test_vector_norm_keepdims_full_reduction() -> None:
    """Return an all-ones TTD when every axis is reduced with keepdims."""
    tensor = arange_tensor(3, 4)
    ttd = TTD(tensor)

    result = np.linalg.vector_norm(ttd, keepdims=True)  # pyright: ignore[reportArgumentType]

    assert isinstance(result, TTD)
    assert result.shape == (1, 1)
    assert_default_epsilon(
        result,
        np.linalg.vector_norm(tensor, keepdims=True),  # pyright: ignore[reportArgumentType]
    )


def test_vector_norm_float32_dtype() -> None:
    """Preserve float32 inputs, matching NumPy."""
    tensor = arange_tensor(3, 4).astype(np.float32)
    ttd = TTD(tensor)

    scalar = ops.vector_norm(ttd)
    reduced = ops.vector_norm(ttd, axis=0)

    assert scalar.dtype == np.float32
    assert isinstance(reduced, TTD)
    assert reduced.dtype == np.float32
    assert ops.vector_norm(ttd, axis=0, keepdims=True).dtype == np.float32
