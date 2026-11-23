# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._helpers import normalize_axes
from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("flip")
def flip[DType: np.floating](
    ttd: TTD[DType], axis: int | Sequence[int] | None = None
) -> TTD[DType]:
    """
    Reverse one or more dimensions of a TTD.

    Equivalent to :func:`numpy.flip` for dense arrays, extended to TTDs. Each
    selected mode is reversed in its core, so the operation is exact and
    preserves the TT-ranks.

    Parameters
    ----------
    ttd : TTD[DType]
        The input TTD.
    axis : int | sequence of int | None, optional
        Axis or axes to reverse. ``None`` reverses every axis. Negative axes
        count from the end. By default None.

    Returns
    -------
    TTD[DType]
        The TTD with the selected axes reversed.

    Raises
    ------
    numpy.exceptions.AxisError
        If an axis is out of range.

    See Also
    --------
    numpy.flip : Equivalent function for dense arrays.
    roll : Roll TTD elements along an axis.
    transpose : Permute the dimensions of a TTD.
    TTD.__getitem__ : Reverse axes with negative-step slices.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> ops.flip(TTD.ones((2, 3))).shape
    (2, 3)

    """
    from wren_ttd.core import TTD

    axes = normalize_axes(axis, ttd.ndim)
    cores = ttd.data.copy()
    for i in axes:
        cores[i] = cores[i][:, ::-1, :]

    return TTD(cores, dtype=ttd.dtype)
