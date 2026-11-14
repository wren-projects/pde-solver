# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy
from typing import cast

import numpy as np
import pytest
from wren_common.tests import TestTensor

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestIndexing:
    """Tests for TTD indexing."""

    def test_full(self, ttd: TestTTD, tensor: TestTensor) -> None:
        """Test full indexing."""
        for index, value in np.ndenumerate(tensor):
            assert_default_epsilon(ttd[index], value)

    def test_single_axis(self, ttd: TestTTD, tensor: TestTensor) -> None:
        """Test indexing along a single axis."""
        for i in range(tensor.shape[0]):
            assert_default_epsilon(ttd[i], tensor[i])
            assert_default_epsilon(ttd[:i], tensor[:i])
            assert_default_epsilon(ttd[i:], tensor[i:])
            assert_default_epsilon(ttd[i::2], tensor[i::2])

    def test_double_axis(self, ttd: TestTTD, tensor: TestTensor) -> None:
        """Test indexing along two axes with integers."""
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                assert_default_epsilon(ttd[i, j], tensor[i, j])
                assert_default_epsilon(cast(TestTTD, ttd[i])[j], tensor[i][j])

    def test_double_axis_slice(self, ttd: TestTTD, tensor: TestTensor) -> None:
        """Test indexing along two axes with slices."""
        for j in range(tensor.shape[1]):
            assert_default_epsilon(ttd[:, j], tensor[:, j])
            n = tensor.shape[0] // 2
            assert_default_epsilon(ttd[:n, j], tensor[:n, j])
            assert_default_epsilon(ttd[:n:2, j], tensor[:n:2, j])
