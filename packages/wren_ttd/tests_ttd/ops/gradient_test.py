# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestGradient:
    """Tests for TTD gradient."""

    def test_default(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient with default arguments."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(np.gradient(ttd), np.gradient(tensor), scale)

    def test_axis_1(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient along a single axis."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(
            np.gradient(ttd, axis=1), np.gradient(tensor, axis=1), scale
        )

    def test_axis_range(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient along a range of axes."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(
            np.gradient(ttd, axis=range(1, ttd.ndim)),
            np.gradient(tensor, axis=range(1, tensor.ndim)),
            scale,
        )

    def test_axis_negative_range(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient along a range of negative axes."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(
            np.gradient(ttd, axis=range(-ttd.ndim + 1, 0)),
            np.gradient(tensor, axis=range(-tensor.ndim + 1, 0)),
            scale,
        )

    def test_uniform_steps(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient with uniform scalar steps."""
        scale = np.linalg.norm(tensor)
        steps = tuple((n + 1) / 10 for n in range(ttd.ndim))
        assert_default_epsilon(
            np.gradient(ttd, *steps), np.gradient(tensor, *steps), scale
        )

    def test_array_steps(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient with array steps."""
        scale = np.linalg.norm(tensor)
        steps = tuple(tuple((x + 1) / 10 for x in range(n)) for n in ttd.shape)
        assert_default_epsilon(
            np.gradient(ttd, *steps), np.gradient(tensor, *steps), scale
        )

    def test_edge_order_2(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient with second-order edge differences."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(
            np.gradient(ttd, edge_order=2), np.gradient(tensor, edge_order=2), scale
        )
