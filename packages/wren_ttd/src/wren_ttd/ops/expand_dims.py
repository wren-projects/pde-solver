# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from wren_ttd._numpy_api import implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("expand_dims")
def expand_dims[DType: np.floating](
    ttd: TTD[DType], axis: int | Sequence[int]
) -> TTD[DType]:
    """
    Insert one or more singleton dimensions into a TTD.

    Equivalent to :func:`numpy.expand_dims` for dense arrays, extended to
    TTDs. New axes are inserted as rank-one identity cores, so the operation
    is exact and preserves the TT-ranks.

    Parameters
    ----------
    ttd : TTD[DType]
        The input TTD.
    axis : int | sequence of int
        Position or positions of the new axes in the result. Negative axes
        count from the end.

    Returns
    -------
    TTD[DType]
        The TTD with the inserted singleton axes.

    Raises
    ------
    ValueError
        If `axis` contains duplicate entries.
    numpy.exceptions.AxisError
        If an axis is out of range for the result.

    See Also
    --------
    numpy.expand_dims : Equivalent function for dense arrays.
    TTD.__getitem__ : Insert singleton axes with ``None`` indexing.
    squeeze : Remove singleton dimensions from a TTD.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> ops.expand_dims(TTD.ones((2, 3)), 1).shape
    (2, 1, 3)

    """
    from wren_ttd.core import TTD

    axes: Sequence[int] = axis if isinstance(axis, Sequence) else (axis,)
    output_ndim = ttd.ndim + len(axes)
    insertion_axes = set(normalize_axis_tuple(axes, output_ndim))

    cores: list[Core[DType]] = []
    source = iter(ttd.data)
    for output_axis in range(output_ndim):
        if output_axis in insertion_axes:
            rank = cores[-1].shape[2] if cores else 1
            cores.append(np.expand_dims(np.eye(rank, dtype=ttd.dtype), 1))
        else:
            cores.append(next(source))

    return TTD(cores, dtype=ttd.dtype)
