from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .swapaxes import swapaxes

_MIN_MATRIX_NDIM = 2


@implements_function("linalg.matrix_transpose")
def matrix_transpose[DType: np.floating](ttd: TTD[DType]) -> TTD[DType]:
    """Transpose the final two dimensions of a TTD."""
    if ttd.ndim < _MIN_MATRIX_NDIM:
        raise ValueError("input array must be at least 2-d")
    return swapaxes(ttd, -1, -2)
