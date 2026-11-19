# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor, arange_tensor
from wren_ttd import ops
from wren_ttd.core import TTD

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(
    ("source_shape", "target_shape"),
    [
        ((3,), (2, 3)),
        ((1, 3), (4, 3)),
        ((2, 1, 3), (2, 4, 3)),
        ((2, 3), (2, 3)),
        ((2, 3), (2, 2, 3)),
        ((1,), (2, 3)),
        ((2, 1, 1), (2, 4, 3)),
    ],
)
def test_broadcast_to(
    source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> None:
    """Test broadcasting through both the ops and NumPy entry points."""
    dense = arange_tensor(*source_shape)
    ttd = TTD(dense)
    expected = np.broadcast_to(dense, target_shape)

    assert_default_epsilon(ops.broadcast_to(ttd, target_shape), expected)
    assert_default_epsilon(np.broadcast_to(ttd, target_shape), expected)


def test_broadcast_to_int_shape() -> None:
    """Test that an int target shape is accepted."""
    dense = arange_tensor(3)
    ttd = TTD(dense)
    assert_default_epsilon(ops.broadcast_to(ttd, 3), dense)
    assert_default_epsilon(np.broadcast_to(ttd, 3), dense)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_broadcast_to_new_leading_axis(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test adding a new leading axis to the shared test tensors."""
    target_shape = (2, *tensor.shape)
    expected = np.broadcast_to(tensor, target_shape)

    assert_default_epsilon(ops.broadcast_to(ttd, target_shape), expected)
    assert_default_epsilon(np.broadcast_to(ttd, target_shape), expected)


@pytest.mark.parametrize(
    ("source_shape", "target_shape"),
    [
        ((2, 3), (4, 3)),
        ((2, 3), (2,)),
        ((3,), (2, 4)),
        ((2, 3), (2, 4, 3)),
    ],
)
def test_broadcast_to_incompatible(
    source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> None:
    """Test that incompatible shapes raise a ValueError like NumPy."""
    display_dense = np.ones(source_shape)
    ttd = TTD(display_dense)

    match = "broadcast|more dimensions"
    with pytest.raises(ValueError, match=match):
        _ = ops.broadcast_to(ttd, target_shape)
    with pytest.raises(ValueError, match=match):
        _ = np.broadcast_to(ttd, target_shape)

    # sanity check that the dense operation fails for the same shapes
    with pytest.raises(ValueError, match=match):
        _ = np.broadcast_to(display_dense, target_shape)


def test_broadcast_to_rejects_subok() -> None:
    """Test that subok=True is rejected."""
    ttd = TTD(arange_tensor(3))
    with pytest.raises(NotImplementedError):
        _ = ops.broadcast_to(ttd, (2, 3), subok=True)
    with pytest.raises(NotImplementedError):
        _ = np.broadcast_to(ttd, (2, 3), subok=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_broadcast_to_preserves_dtype(dtype: type[np.floating]) -> None:
    """Test that the result keeps the input dtype."""
    ttd = TTD(arange_tensor(1, 3).astype(dtype))
    assert ops.broadcast_to(ttd, (4, 3)).dtype == dtype
    assert np.broadcast_to(ttd, (4, 3)).dtype == dtype
