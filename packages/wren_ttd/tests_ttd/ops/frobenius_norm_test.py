# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_frobenius_norm(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test Frobenius norm."""
    assert_default_epsilon(np.linalg.norm(ttd), np.linalg.norm(tensor))
