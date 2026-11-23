# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import ops
from wren_ttd.core import TTD

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_norm(tensor: TestTensor, ttd: TestTTD) -> None:
    """Compare the default norm against NumPy."""
    assert_default_epsilon(ops.norm(ttd), np.linalg.norm(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("axis", [0, None])
def test_norm_axis(tensor: TestTensor, ttd: TestTTD, axis: int | None) -> None:
    """Compare norms along an axis against NumPy."""
    assert_default_epsilon(
        ops.norm(ttd, axis=axis),
        np.linalg.norm(tensor, axis=axis),
    )


def test_norm_keepdims() -> None:
    """Keep reduced axes when requested."""
    ttd = ops.norm(TTD.ones((2, 3)), axis=1, keepdims=True)

    assert ttd.shape == (2, 1)


@pytest.mark.parametrize("order", [1, -1, np.inf, "nuc"])
def test_norm_rejects_unsupported_orders(order: float | str) -> None:
    """Reject norm orders outside the Euclidean and Frobenius norms."""
    with pytest.raises(NotImplementedError, match="Euclidean"):
        _ = ops.norm(TTD.ones((2, 2)), ord=order)
