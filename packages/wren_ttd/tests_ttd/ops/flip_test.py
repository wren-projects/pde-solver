# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("axis", [None, 0, -1, (0, -1), ()])
def test_flip(
    tensor: TestTensor, ttd: TestTTD, axis: int | tuple[int, ...] | None
) -> None:
    """Test flip via ops and NumPy."""
    expected = np.flip(tensor, axis=axis)
    assert_default_epsilon(ops.flip(ttd, axis), expected)
    assert_default_epsilon(np.flip(ttd, axis=axis), expected)
