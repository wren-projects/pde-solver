# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD[:3]))
def test_astype(tensor: TestTensor, ttd: TestTTD) -> None:
    """Cast through the method, ops, and NumPy Array API."""
    expected = tensor.astype(np.float32)
    casted: TTD[np.float32] = ttd.astype(np.float32)
    ops_casted: TTD[np.float32] = ops.astype(ttd, np.float32)
    namespace_casted = cast(
        TTD[np.float32], cast(Any, np.astype(cast(Any, ttd), np.float32))
    )
    assert isinstance(namespace_casted, TTD)

    for result in (casted, ops_casted, namespace_casted):
        assert result.dtype == np.dtype(np.float32)
        assert all(core.dtype == np.dtype(np.float32) for core in result.data)
        assert_default_epsilon(result, expected, scale=tensor, epsilon=1e-5)


@pytest.mark.parametrize(("_tensor", "ttd"), deepcopy(TEST_TTD[:3]))
def test_astype_doesnt_copy_same_dtype(_tensor: TestTensor, ttd: TestTTD) -> None:
    """copy=False doesn't copy the same dtype."""
    no_copied = ttd.astype(ttd.dtype, copy=False)
    assert no_copied is ttd
    assert all(
        new_core is core
        for new_core, core in zip(no_copied.data, ttd.data, strict=True)
    )


def test_astype_rejects_non_floating_dtype() -> None:
    """Reject target dtypes unsupported by the TTD representation."""
    with pytest.raises(TypeError, match="not a valid dtype for TTD"):
        _ = TTD.ones((2, 3)).astype(np.int64)
