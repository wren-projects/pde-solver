# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_absolute(tensor: TestTensor, ttd: TestTTD) -> None:
    """Compare the element-wise absolute value against NumPy."""
    assert_default_epsilon(np.absolute(ttd), np.absolute(tensor))
    assert_default_epsilon(abs(ttd), abs(tensor))
    assert_default_epsilon(ops.absolute(ttd), np.absolute(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_positive(tensor: TestTensor, ttd: TestTTD) -> None:
    """Compare the unary plus operator against NumPy."""
    assert_default_epsilon(np.positive(ttd), np.positive(tensor))
    assert_default_epsilon(+ttd, +tensor)
    assert_default_epsilon(ops.positive(ttd), np.positive(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_power(tensor: TestTensor, ttd: TestTTD) -> None:
    """Compare integer powers against NumPy."""
    assert_default_epsilon(np.power(ttd, 2), np.power(tensor, 2))
    assert_default_epsilon(ttd**3, tensor**3)
    assert_default_epsilon(ops.power(ttd, 2), np.power(tensor, 2))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_round(tensor: TestTensor, ttd: TestTTD) -> None:
    """Compare element-wise rounding against NumPy."""
    assert_default_epsilon(np.round(ttd, 2), np.round(tensor, 2))
    assert_default_epsilon(np.around(ttd, 2), np.around(tensor, 2))
    assert_default_epsilon(ops.round_(ttd, 2), np.round(tensor, 2))


def test_power_rejects_fractional_exponent() -> None:
    """Reject non-integer exponents instead of densifying or returning NaN."""
    ttd = TTD.ones((2, 2))

    with pytest.raises(NotImplementedError, match="integer-like"):
        _ = np.power(ttd, 0.5)


def test_power_rejects_ttd_exponent() -> None:
    """Reject TTD exponents, which are not supported."""
    ttd = TTD.ones((2, 2))

    with pytest.raises(NotImplementedError, match="integer-like"):
        _ = np.power(ttd, ttd)
