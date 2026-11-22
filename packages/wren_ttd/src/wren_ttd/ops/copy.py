# ruff: noqa: PLC0415, FBT001, FBT002
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("copy")
def copy[DType: np.floating](
    array: TTD[DType], order: str = "K", subok: bool = False
) -> TTD[DType]:
    """Return a copy of a TTD."""
    from wren_ttd.core import TTD

    if order not in ("C", "F", "A", "K"):
        raise ValueError(f"order must be one of 'C', 'F', 'A', or 'K', got {order!r}")

    cores = [core.copy() for core in array.data]
    if subok:
        return array.__class__(cores)
    return TTD(cores, dtype=array.dtype)
