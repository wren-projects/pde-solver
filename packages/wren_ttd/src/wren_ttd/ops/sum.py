# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple
from wren_common.math import dot_product
from wren_common.types import Matrix

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("sum")
def sum[DType: np.floating](  # noqa: A001
    ttd: TTD[DType], axis: int | Sequence[int] | None = None
) -> DType | TTD[DType]:
    """
    Return the sum of the TTD object along the given axis.

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD object to sum.
    axis : int | Sequence[int] | None, optional
        The axis or axes along which to sum. If None, sum all axes.

    Returns
    -------
    TTD[DType]
        The result of the sum.

    """
    from wren_ttd.core import TTD

    if axis is None:
        axis = tuple(range(ttd.ndim))

    axes = normalize_axis_tuple(axis, ttd.ndim)

    cores = ttd.data.copy()

    for a in sorted(axes, reverse=True):
        if len(cores) == 1:
            return ttd.dtype.type(cast(np.floating, np.sum(cores[0], axis=1)))

        summed = cast(Matrix[DType], np.sum(cores[a], axis=1))

        if a == 0:
            cores[1] = dot_product(summed, cores[1])
        else:
            cores[a - 1] = dot_product(cores[a - 1], summed)

        _ = cores.pop(a)

    return TTD(cores, dtype=ttd.dtype)
