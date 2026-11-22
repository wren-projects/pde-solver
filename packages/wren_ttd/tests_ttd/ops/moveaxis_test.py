# pyright: reportAny=false, reportUnknownVariableType=false
from typing import Any, cast

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), TEST_TTD[1:4])
@pytest.mark.parametrize(
    ("source", "destination"),
    [
        (0, -1),
        (-1, 0),
        ((0, 2), (2, 0)),
        ((0, 1), (-1, -2)),
        ((), ()),
    ],
)
def test_moveaxis(
    tensor: TestTensor,
    ttd: TestTTD,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
) -> None:
    """Compare TTD moveaxis with NumPy for scalar and multiple axes."""
    expected = np.moveaxis(tensor, source, destination)
    assert_default_epsilon(ops.moveaxis(ttd, source, destination), expected)
    moved = cast(np.ndarray[Any, Any], np.moveaxis(ttd, source, destination))  # pyright: ignore[reportArgumentType]
    assert_default_epsilon(moved, expected)


@pytest.mark.parametrize(
    ("source", "destination", "message"),
    [
        ((0, 1), (2,), "same number of axes"),
        ((0, 0), (1, 2), "repeated axis"),
    ],
)
def test_moveaxis_rejects_invalid_axes(
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
    message: str,
) -> None:
    """Reject invalid source/destination axis specifications."""
    ttd = TTD(np.ones((2, 3, 4)))
    with pytest.raises(ValueError, match=message):
        _ = ops.moveaxis(ttd, source, destination)
