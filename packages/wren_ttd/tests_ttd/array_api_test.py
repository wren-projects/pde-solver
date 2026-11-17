# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensor
from wren_ttd import TTD

from .common import TEST_TTD, TestTTD, assert_default_epsilon


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_array_namespace(tensor: TestTensor, ttd: TestTTD) -> None:
    """Return NumPy's Array API-compatible namespace."""
    namespace = ttd.__array_namespace__()

    assert namespace is np
    assert namespace.__array_api_version__ == np.__array_api_version__
    assert_default_epsilon(namespace.add(ttd, ttd), tensor * 2, scale=tensor)


def test_array_namespace_versions() -> None:
    """Accept versions supported by NumPy's Array API namespace."""
    ttd = TTD.ones((2, 3))

    assert ttd.__array_namespace__(api_version="2023.12") is np
    assert ttd.__array_namespace__(api_version=np.__array_api_version__) is np


def test_array_namespace_rejects_unsupported_version() -> None:
    """Reject Array API versions unsupported by NumPy."""
    with pytest.raises(ValueError, match="not supported"):
        _ = TTD.ones((2, 3)).__array_namespace__(api_version="unsupported")
