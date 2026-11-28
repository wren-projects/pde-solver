# pyright: reportAny=false, reportUnknownVariableType=false
import numpy as np
import pytest
from wren_common.tests import arange_tensor
from wren_ttd import TTD, ops

from ..common import assert_default_epsilon


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (arange_tensor(3), arange_tensor(3)),
        (arange_tensor(2, 3), arange_tensor(3)),
        (arange_tensor(3), arange_tensor(3, 2)),
        (arange_tensor(2, 3), arange_tensor(3, 4)),
        (
            arange_tensor(2, 3),
            arange_tensor(2, 3, 4),
        ),
        (
            arange_tensor(2, 3, 4),
            arange_tensor(4, 5),
        ),
        (
            arange_tensor(2, 3, 4),
            arange_tensor(4),
        ),
        (
            arange_tensor(4),
            arange_tensor(2, 4, 3),
        ),
        (
            arange_tensor(2, 2, 3),
            arange_tensor(2, 3, 2),
        ),
        (
            arange_tensor(2, 3, 3, 4),
            arange_tensor(2, 3, 4, 3),
        ),
    ],
)
def test_matmul(a: np.ndarray, b: np.ndarray) -> None:
    """Compare supported TTD operand ranks against NumPy."""
    ttd_a, ttd_b = TTD(a), TTD(b)
    assert_default_epsilon(ops.matmul(ttd_a, ttd_b), np.matmul(a, b))
    assert_default_epsilon(np.matmul(ttd_a, ttd_b), np.matmul(a, b))


def test_matrix_multiply_operator() -> None:
    """Support Python's matrix multiplication operator."""
    a, b = arrays = arange_tensor(2, 3), arange_tensor(3, 4)
    ttd_a, ttd_b = map(TTD, arrays)

    assert_default_epsilon(ttd_a @ ttd_b, a @ b)


def test_matrix_multiply_operator_inplace() -> None:
    """Support Python's matrix multiplication operator."""
    a, b = arrays = arange_tensor(3, 3), 5 * arange_tensor(3, 3)
    ttd_a, ttd_b = map(TTD, arrays)

    ttd_a @= ttd_b

    assert_default_epsilon(ttd_a, a @ b)
