from copy import deepcopy

import numpy as np
import pytest
from wren_ttd import TTD, ops

from ..common import TEST_TTD, TestTTD


@pytest.mark.parametrize("ttd", [ttd for _, ttd in deepcopy(TEST_TTD)])
def test_to_device_cpu_is_noop(ttd: TestTTD) -> None:
    """The CPU transfer operation returns the same TTD."""
    assert ttd.device == "cpu"
    assert ttd.to_device("cpu") is ttd
    assert ops.to_device(ttd, "cpu") is ttd


def test_to_device_rejects_unsupported_device_and_stream() -> None:
    """Reject non-CPU devices and unsupported streams."""
    ttd: TTD[np.float64] = TTD.ones((2, 3))
    with pytest.raises(ValueError, match="CPU"):
        _ = ops.to_device(ttd, "gpu")
    with pytest.raises(NotImplementedError, match="streams"):
        _ = ops.to_device(ttd, "cpu", stream=object())
