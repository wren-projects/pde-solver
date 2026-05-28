# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
import pytest
from wren_common.tests import TestTensor

from ..common import (
    TEST_TTD,
    TestTTD,
    assert_default_epsilon,
)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestSum:
    """Tests for TTD summation."""

    def test_full(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test summation over all axes."""
        assert_default_epsilon(np.sum(ttd), np.sum(tensor))

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_single_axis(self, tensor: TestTensor, ttd: TestTTD, axis: int) -> None:
        """Test summation over a single axis."""
        assert_default_epsilon(ttd.sum(axis), cast(TestTensor, tensor.sum(axis)))

    def test_axes_tuple(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test summation over a tuple of axes."""
        assert_default_epsilon(ttd.sum((0, 1)), cast(TestTensor, tensor.sum((0, 1))))
