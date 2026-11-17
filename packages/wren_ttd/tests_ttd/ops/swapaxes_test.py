# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize(
    "axes",
    [(0, 1), (0, -1), (-1, -2), (0, 0)],
)
class TestSwapaxes:
    """Tests for TTD swapaxes."""

    def test_numpy(
        self, tensor: TestTensor, ttd: TestTTD, axes: tuple[int, int]
    ) -> None:
        """Test swapaxes via ``np.swapaxes``."""
        axis1, axis2 = axes
        assert_default_epsilon(
            np.swapaxes(ttd, axis1, axis2), np.swapaxes(tensor, axis1, axis2)
        )

    def test_method(
        self, tensor: TestTensor, ttd: TestTTD, axes: tuple[int, int]
    ) -> None:
        """Test swapaxes via the ``swapaxes`` method."""
        axis1, axis2 = axes
        assert_default_epsilon(
            ttd.swapaxes(axis1, axis2), tensor.swapaxes(axis1, axis2)
        )
