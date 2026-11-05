# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TEST_SCALARS, TestTensor

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("value", TEST_SCALARS)
@pytest.mark.parametrize("width", [0, 1, 2])
class TestPad:
    """Tests for TTD padding."""

    def test_pad_single_constant(
        self, tensor: TestTensor, ttd: TestTTD, value: float, width: int
    ) -> None:
        """Test pad with the selected constant value in single width."""
        expected = np.pad(tensor, width, mode="constant", constant_values=value)
        result = np.pad(ttd, width, mode="constant", constant_values=value)
        assert_default_epsilon(result, expected, scale=value)
