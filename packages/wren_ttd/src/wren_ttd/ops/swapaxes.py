from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .transpose import transpose


@implements_function("swapaxes")
def swapaxes[DType: np.floating](ttd: TTD[DType], axis1: int, axis2: int) -> TTD[DType]:
    """
    Swap the axes of a TTD tensor.

    See ::func:`transpose` for more details.

    Parameters
    ----------
    ttd : TTD[DType]
        Input TTD tensor.
    axis1 : int
        First axis to swap.
    axis2 : int
        Second axis to swap.

    Returns
    -------
    TTD[DType]
        TTD tensor with `axis1` and `axis2` swapped.

    """
    axis1 = normalize_axis_index(axis1, ttd.ndim)
    axis2 = normalize_axis_index(axis2, ttd.ndim)

    if axis1 == axis2:
        return ttd

    axes = list(range(ttd.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

    return transpose(ttd, axes)
