# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import pytest
from wren_common.tests import TestTensorPair, arange_tensor
from wren_ttd import TTD, ops

from ..common import TEST_PAIR_TTD, TestTTDPair, assert_default_epsilon


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
def test_vecdot_default_axis(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Compare the default last-axis dot product against NumPy."""
    a, b = tensors
    ttd_a, ttd_b = ttds

    assert_default_epsilon(np.vecdot(ttd_a, ttd_b), np.vecdot(a, b))
    assert_default_epsilon(ops.vecdot(ttd_a, ttd_b), np.vecdot(a, b))


def test_vecdot_vector() -> None:
    """Reduce a one-dimensional pair to a scalar."""
    a = arange_tensor(4)
    b = np.array([-1.0, 2.0, 0.5, 3.0])
    ttd_a, ttd_b = TTD(a), TTD(b)

    assert_default_epsilon(np.vecdot(ttd_a, ttd_b), np.vecdot(a, b))
    assert_default_epsilon(ops.vecdot(ttd_a, ttd_b), np.vecdot(a, b))
    assert isinstance(np.vecdot(ttd_a, ttd_b), np.floating)
    assert isinstance(ops.vecdot(ttd_a, ttd_b), np.floating)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((3, 4), 0),
        ((3, 4), -2),
        ((2, 3, 4), 1),
        ((2, 3, 4), -3),
    ],
)
def test_vecdot_explicit_axis(shape: tuple[int, ...], axis: int) -> None:
    """Compare explicit and negative axes against NumPy."""
    a = arange_tensor(*shape)
    b = arange_tensor(*shape) + 1.0
    ttd_a, ttd_b = TTD(a), TTD(b)
    expected = np.vecdot(a, b, axis=axis)  # pyright: ignore[reportCallIssue]

    assert_default_epsilon(
        np.vecdot(ttd_a, ttd_b, axis=axis),  # pyright: ignore[reportCallIssue]
        expected,
    )
    assert_default_epsilon(ops.vecdot(ttd_a, ttd_b, axis=axis), expected)


def test_vecdot_promotes_dtypes() -> None:
    """Promote operand dtypes with NumPy's rules."""
    a = TTD(arange_tensor(3, 4), dtype=np.float32)
    b = cast(TTD[np.float32], TTD(arange_tensor(3, 4)))

    result = ops.vecdot(a, b)

    assert isinstance(result, TTD)
    assert result.dtype == np.float64


def test_vecdot_rejects_broadcasting() -> None:
    """Reject operands whose shapes differ instead of broadcasting."""
    a = TTD.ones((2, 3))
    b = TTD.ones((2, 4))

    with pytest.raises(NotImplementedError, match="broadcasting"):
        _ = np.vecdot(a, b)

    with pytest.raises(NotImplementedError, match="broadcasting"):
        _ = ops.vecdot(a, b)


def test_vecdot_result_shapes() -> None:
    """Preserve the unreduced axes and remove the contracted one."""
    a = TTD(arange_tensor(2, 3, 4))
    b = TTD(arange_tensor(2, 3, 4))

    last = ops.vecdot(a, b, axis=-1)
    first = ops.vecdot(a, b, axis=0)

    assert isinstance(last, TTD)
    assert last.shape == (2, 3)
    assert isinstance(first, TTD)
    assert first.shape == (3, 4)
