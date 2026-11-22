# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("axis", [0, 1, -1, (0, 2), (1, 3)])
def test_expand_dims(
    tensor: TestTensor, ttd: TestTTD, axis: int | tuple[int, ...]
) -> None:
    """Test expand_dims via ops and NumPy."""
    expected = np.expand_dims(tensor, axis=axis)
    assert_default_epsilon(ops.expand_dims(ttd, axis), expected)
    assert_default_epsilon(np.expand_dims(ttd, axis=axis), expected)
