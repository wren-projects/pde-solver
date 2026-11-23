# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false
from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from wren_common.tests import arange_tensor
from wren_ttd import TTD, ops
from wren_ttd._numpy_api import DenseWarning


def _assert_no_dense_warning(operation: Callable[[], object]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DenseWarning)
        _ = operation()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ops.absolute(TTD(arange_tensor(3))),
        lambda: np.round(TTD(arange_tensor(3)), 1),
        lambda: ops.power(TTD.ones(3), -1),
        lambda: ops.reshape(TTD(arange_tensor(2, 3)), 6),
        lambda: ops.ravel(TTD(arange_tensor(2, 3))),
        lambda: ops.flatten(TTD(arange_tensor(2, 3))),
        lambda: ops.cumsum(TTD(arange_tensor(2, 3))),
        lambda: ops.take(TTD(arange_tensor(2, 3)), [0, 1]),
        lambda: ops.repeat(TTD(arange_tensor(2, 3)), 2),
        lambda: ops.concat(
            [TTD(arange_tensor(2, 3)), TTD(arange_tensor(2, 3))], axis=None
        ),
        lambda: ops.vector_norm(TTD(arange_tensor(2, 3)), axis=0),
        lambda: ops.matrix_norm(TTD(arange_tensor(2, 3, 4))),
    ],
)
def test_densifying_operations_warn(operation: Callable[[], object]) -> None:
    """Warn whenever an operation expands a TTD to a dense array."""
    with pytest.warns(DenseWarning, match="densif"):
        _ = operation()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ops.cumsum(TTD(arange_tensor(2, 3))),
        lambda: ops.take(TTD(arange_tensor(2, 3)), [0, 1]),
        lambda: ops.repeat(TTD(arange_tensor(2, 3)), 2),
        lambda: ops.concat(
            [TTD(arange_tensor(2, 3)), TTD(arange_tensor(2, 3))], axis=None
        ),
        lambda: ops.vector_norm(TTD(arange_tensor(2, 3)), axis=0),
        lambda: ops.matrix_norm(TTD(arange_tensor(2, 3, 4))),
    ],
)
def test_nested_dense_warning_points_to_the_caller(
    operation: Callable[[], object],
) -> None:
    """Attribute warnings from delegated operations to the caller."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = operation()

    dense = [
        warning for warning in caught if issubclass(warning.category, DenseWarning)
    ]
    assert dense
    assert Path(dense[0].filename).resolve() == Path(__file__).resolve()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ops.power(TTD.ones(3), 2),
        lambda: ops.reshape(TTD(arange_tensor(2, 3)), (3, 2)),
        lambda: ops.ravel(TTD(arange_tensor(3))),
        lambda: ops.flatten(TTD(arange_tensor(3))),
        lambda: ops.cumsum(TTD(arange_tensor(2, 3)), axis=0),
        lambda: ops.take(TTD(arange_tensor(2, 3)), [0, 1], axis=0),
        lambda: ops.repeat(TTD(arange_tensor(2, 3)), 2, axis=0),
        lambda: ops.concat(
            [TTD(arange_tensor(2, 3)), TTD(arange_tensor(2, 3))], axis=0
        ),
        lambda: ops.vector_norm(TTD(arange_tensor(2, 3))),
        lambda: ops.matrix_norm(TTD(arange_tensor(2, 3))),
    ],
)
def test_exact_operations_do_not_warn(operation: Callable[[], object]) -> None:
    """Do not warn when the operation stays in the TT representation."""
    _assert_no_dense_warning(operation)
