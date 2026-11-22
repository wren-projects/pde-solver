# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from numpy.lib.array_utils import normalize_axis_index

from wren_ttd._numpy_api import implements_function
from wren_ttd.ops.ravel import ravel

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("repeat")
def repeat[DType: np.floating](
    ttd: TTD[DType],
    repeats: int | Sequence[int] | npt.NDArray[np.integer[Any]],
    axis: int | None = None,
) -> TTD[DType]:
    """
    Repeat elements of a TTD.

    Equivalent to :func:`numpy.repeat` for dense arrays, extended to TTDs.
    The operation is exact and preserves the TT-ranks of the input.

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD to repeat elements of.
    repeats : int | sequence of int
        The number of repetitions for each element. If a scalar, every
        element is repeated that many times. If a sequence, it must have
        length ``ttd.shape[axis]`` and gives the repetitions per element.
        Negative repetitions are an error.
    axis : int | None, optional
        The axis along which to repeat values. May be negative. If ``None``,
        the input is flattened first, by default ``None``.

    Returns
    -------
    TTD[DType]
        The repeated TTD, with the selected axis replaced by the repeated
        indices.

    Raises
    ------
    ValueError
        If `repeats` is negative or has the wrong length, or if `axis` is
        out of range.

    See Also
    --------
    numpy.repeat : Equivalent function for dense arrays.
    tile : Construct a TTD by repeating it.
    take : Take elements from a TTD along an axis.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 3))
    >>> ops.repeat(a, 2, axis=0).shape
    (4, 3)
    >>> ops.repeat(a, [1, 2, 0], axis=1).shape
    (2, 3)
    >>> ops.repeat(a, 2).shape
    (12,)

    """
    from wren_ttd.core import TTD

    if axis is None:
        axis = 0
        if ttd.ndim > 1:
            ttd = ravel(ttd)

    axis = normalize_axis_index(axis, ttd.ndim)

    cores = ttd.data.copy()
    cores[axis] = np.repeat(cores[axis], repeats, axis=1)
    return TTD(cores, dtype=ttd.dtype)
