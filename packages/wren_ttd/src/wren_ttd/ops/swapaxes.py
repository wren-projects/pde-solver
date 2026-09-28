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
    Swap two axes of a TTD tensor.

    Equivalent to :func:`numpy.swapaxes` for dense arrays, extended to
    TTDs. Implemented in terms of :func:`transpose`; see it for more
    details.

    If `axis1` and `axis2` are equal, the TTD is returned unchanged.

    Parameters
    ----------
    ttd : TTD[DType]
        Input TTD tensor.
    axis1 : int
        First axis to swap. May be negative, counting from the end.
    axis2 : int
        Second axis to swap. May be negative, counting from the end.

    Returns
    -------
    TTD[DType]
        TTD tensor with `axis1` and `axis2` swapped.

    See Also
    --------
    numpy.swapaxes : Equivalent function for dense arrays.
    transpose : Permute the dimensions of a TTD.
    TTD.swapaxes : Swap two axes of a TTD.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 3, 4))
    >>> ops.swapaxes(t, 0, 2).shape
    (4, 3, 2)

    """
    axis1 = normalize_axis_index(axis1, ttd.ndim)
    axis2 = normalize_axis_index(axis2, ttd.ndim)

    if axis1 == axis2:
        return ttd

    axes = list(range(ttd.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

    return transpose(ttd, axes)
