# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize(
    ("shift", "axis"),
    [(0, 0), (1, 0), (-1, -1), (2, 0), (1, (0, -1)), ((1, -2), (0, -1)), (3, ())],
)
def test_roll(
    tensor: TestTensor,
    ttd: TestTTD,
    shift: int | tuple[int, ...],
    axis: int | tuple[int, ...],
) -> None:
    """Test roll via ops and NumPy."""
    expected = np.roll(tensor, shift, axis=axis)
    assert_default_epsilon(ops.roll(ttd, shift, axis), expected)
    assert_default_epsilon(np.roll(ttd, shift, axis=axis), expected)


def test_roll_flattened_not_supported() -> None:
    """Test that rolling over the flattened tensor raises."""
    _, ttd = deepcopy(TEST_TTD[0])
    with pytest.raises(NotImplementedError):
        _ = ops.roll(ttd, 1)
