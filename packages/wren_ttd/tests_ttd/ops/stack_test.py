# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor, TestTensorPair

from ..common import (
    TEST_PAIR_TTD,
    TEST_TTD,
    TestTTD,
    TestTTDPair,
    assert_default_epsilon,
)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestStackSingle:
    """Tests for stacking a single TTD."""

    def test_stack(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test stacking a single tensor."""
        assert_default_epsilon(
            np.stack(ttd),
            np.stack(list(tensor)),
        )


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
class TestStack:
    """Tests for stacking TTD pairs."""

    @pytest.mark.parametrize("axis", [0, -1, 1])
    @pytest.mark.parametrize("repeat", [1, 2])
    def test_stack(
        self, tensors: TestTensorPair, ttds: TestTTDPair, axis: int, repeat: int
    ) -> None:
        """Test stack for the selected axis, with optional repetition."""
        assert_default_epsilon(
            np.stack(ttds * repeat, axis=axis),
            np.stack(tensors * repeat, axis=axis),
        )
