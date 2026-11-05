# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestTranspose:
    """Tests for TTD transpose."""

    def test_default_method(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose via the ``transpose`` method."""
        assert_default_epsilon(ttd.transpose(), tensor.transpose())

    def test_property_t(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose via the ``T`` property."""
        assert_default_epsilon(ttd.T, tensor.T)

    def test_swap_first_two(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose swapping the first two axes."""
        axes = (1, 0, *range(2, tensor.ndim))
        assert_default_epsilon(
            np.transpose(ttd, axes),
            np.transpose(tensor, axes),
        )

    def test_rotate(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose rotating axes to the left."""
        axes = (*range(1, tensor.ndim), 0)
        assert_default_epsilon(
            np.transpose(ttd, axes),
            np.transpose(tensor, axes),
        )

    def test_last_to_front(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose moving the last axis to the front."""
        axes = (-1, *range(tensor.ndim - 1))
        assert_default_epsilon(
            np.transpose(ttd, axes),
            np.transpose(tensor, axes),
        )

    def test_reverse(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose reversing all axes."""
        axes = tuple(reversed(range(tensor.ndim)))
        assert_default_epsilon(
            np.transpose(ttd, axes),
            np.transpose(tensor, axes),
        )
