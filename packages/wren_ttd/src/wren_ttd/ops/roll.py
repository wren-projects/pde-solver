# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._helpers import normalize_axes
from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("roll")
def roll[DType: np.floating](
    ttd: TTD[DType],
    shift: int | Sequence[int],
    axis: int | Sequence[int] | None = None,
) -> TTD[DType]:
    """
    Roll the elements of a TTD along one or more axes.

    This is the TTD equivalent of :func:`numpy.roll`. Rolling along an axis is
    a cyclic permutation of that axis' indices, so it is applied exactly by
    permuting the indices of the corresponding core and the TT ranks are kept.

    Parameters
    ----------
    ttd : TTD[DType]
        Input tensor.
    shift : int or sequence of ints
        Number of places each element is shifted. A sequence must match the
        number of axes in `axis`.
    axis : int or sequence of ints
        Axis or axes along which elements are shifted. Each axis must be unique.
        ``None`` (flattened rolling) is not supported.

    Returns
    -------
    TTD[DType]
        Rolled tensor with the same shape as the input.

    Raises
    ------
    NotImplementedError
        If `axis` is None.
    ValueError
        If an axis is out of range or repeated, or if `shift` does not match
        `axis`.

    See Also
    --------
    numpy.roll : Equivalent function for dense arrays.
    flip : Reverse the order of elements along the given axes.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 3, 4))
    >>> ops.roll(t, 1, axis=1).shape
    (2, 3, 4)

    """
    from wren_ttd.core import TTD

    if axis is None:
        raise NotImplementedError(
            "Rolling over a flattened TTD (axis=None) is not supported"
        )

    axes = normalize_axes(axis, ttd.ndim)
    shifts = [shift] * len(axes) if isinstance(shift, int) else shift

    cores = ttd.data.copy()
    for i, s in zip(axes, shifts, strict=True):
        cores[i] = np.roll(cores[i], s, axis=1)

    return TTD(cores, dtype=ttd.dtype)
