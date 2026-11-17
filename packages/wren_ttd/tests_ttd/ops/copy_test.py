# pyright: reportAny=false, reportUnknownVariableType=false
from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_copy(tensor: TestTensor, ttd: TestTTD) -> None:
    """Copy through the method, ops, and NumPy."""
    copies = (ttd.copy(), ops.copy(ttd), np.copy(ttd))
    for copied in copies:
        assert isinstance(copied, TTD)
        assert copied is not ttd
        assert all(a is not b for a, b in zip(copied.data, ttd.data, strict=True))
        assert_default_epsilon(copied, tensor)

    copy_with_subok = cast(Callable[..., Any], np.copy)
    numpy_subclass_copy = copy_with_subok(cast(Any, ttd), subok=True)
    assert isinstance(numpy_subclass_copy, TTD)
    assert_default_epsilon(cast(TestTTD, numpy_subclass_copy), tensor)
