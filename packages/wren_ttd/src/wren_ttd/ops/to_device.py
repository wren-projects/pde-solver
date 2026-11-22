from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def to_device[DType: np.floating](
    array: TTD[DType], device: str, /, *, stream: object | None = None
) -> TTD[DType]:
    """Return a TTD on the requested device."""
    if device != "cpu":
        raise ValueError("TTD arrays are only supported on the CPU")
    if stream is not None:
        raise NotImplementedError("TTD device transfers do not support streams")
    return array
