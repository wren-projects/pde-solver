# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensorPair

from ..common import TEST_PAIR_TTD, TestTTDPair, assert_default_epsilon


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
def test_inner_product(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test inner product."""
    ttd_a, ttd_b = ttds
    tensor_a, tensor_b = tensors

    assert_default_epsilon(np.vdot(ttd_a, ttd_b), np.vdot(tensor_a, tensor_b))
