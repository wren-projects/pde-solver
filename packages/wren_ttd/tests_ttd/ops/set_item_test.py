# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import pytest
from wren_common.tests import TEST_SCALARS, TestTensor
from wren_common.types import Scalar

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize(
    "index",
    [
        0,
        2,
        -1,
        slice(1, -1),
        slice(None, None, 2),
        slice(1, None, 2),
        slice(None, None, -1),
    ],
)
@pytest.mark.parametrize("value", TEST_SCALARS)
def test_setitem(
    tensor: TestTensor, ttd: TestTTD, index: int | slice, value: Scalar
) -> None:
    """Test zeroing a single mode addressed by an integer or a slice."""
    for axis, _ in enumerate(tensor.shape):
        leading = (slice(None),) * axis
        trailing = (slice(None),) * (len(tensor.shape) - axis - 1)
        key = (*leading, index, *trailing)

        expected = tensor.copy()
        expected[key] = value

        result = ttd.copy()
        result[key] = value

        assert_default_epsilon(result, expected, scale=value)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_setitem_boundary(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test repeatedly zeroing boundary slices of a TTD."""
    expected = tensor.copy()
    result = ttd.copy()

    for axis, length in enumerate(tensor.shape):
        for index in (0, length - 1):
            key: list[int | slice] = [slice(None)] * tensor.ndim
            key[axis] = index
            expected[tuple(key)] = 0
            result[tuple(key)] = 0

    assert_default_epsilon(result, expected)
