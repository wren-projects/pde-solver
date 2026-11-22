from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .transpose import transpose


@implements_function("moveaxis")
def moveaxis[DType: np.floating](
    ttd: TTD[DType],
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> TTD[DType]:
    """
    Move axes of a TTD to new positions.

    This is the TTD equivalent of :func:`numpy.moveaxis`. The axes listed in
    `source` are moved to their respective positions in `destination`; the
    relative order of all other axes is unchanged. The relative order among
    the moved axes is also preserved.

    Parameters
    ----------
    ttd : TTD[DType]
        Input tensor.
    source : int or sequence of ints
        Original positions of the axes to move. Each axis must be unique and
        in the range ``[-ttd.ndim, ttd.ndim)``. Negative indices count from
        the end.
    destination : int or sequence of ints
        Destination positions for the moved axes, in the same order as
        `source`. Must contain the same number of axes. Negative indices count
        from the end.

    Returns
    -------
    TTD[DType]
        Tensor with the requested axes moved.

    Raises
    ------
    ValueError
        If an axis is out of range, is repeated, or if `source` and
        `destination` contain different numbers of axes.

    See Also
    --------
    numpy.moveaxis : Equivalent function for dense arrays.
    transpose : Permute all tensor axes.
    swapaxes : Swap two tensor axes.

    Notes
    -----
    The operation constructs an axis permutation and applies
    :func:`transpose`. As with other nontrivial TTD transposes, the result may
    be approximate because adjacent core swaps use TTD-SVD truncation.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 3, 4))
    >>> ops.moveaxis(t, 0, -1).shape
    (3, 4, 2)
    >>> ops.moveaxis(t, (0, 1), (-1, -2)).shape
    (4, 3, 2)

    """
    source_axes = normalize_axis_tuple(source, ttd.ndim, "source")
    destination_axes = normalize_axis_tuple(destination, ttd.ndim, "destination")

    if len(source_axes) != len(destination_axes):
        raise ValueError("source and destination must have the same number of axes")

    permutation = [axis for axis in range(ttd.ndim) if axis not in source_axes]
    for destination_axis, source_axis in sorted(
        zip(destination_axes, source_axes, strict=True)
    ):
        permutation.insert(destination_axis, source_axis)

    return transpose(ttd, permutation)
