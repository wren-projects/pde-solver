# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .get_item import get_item


@implements_function("squeeze")
def squeeze[DType: np.floating](
    ttd: TTD[DType], axis: int | Sequence[int] | None = None
) -> TTD[DType] | DType:
    """
    Remove singleton dimensions from a TTD.

    Equivalent to :func:`numpy.squeeze` for dense arrays, extended to TTDs.
    Without `axis`, every mode of size one is removed. If every mode is
    removed, the result is a NumPy scalar because TTD has no rank-zero
    representation.

    Parameters
    ----------
    ttd : TTD[DType]
        The input TTD.
    axis : int | sequence of int | None, optional
        Axis or axes to remove. ``None`` removes every size-one axis.
        Negative axes count from the end. By default None.

    Returns
    -------
    TTD[DType] | DType
        The squeezed TTD, or a NumPy scalar if all axes are removed.

    Raises
    ------
    ValueError
        If a selected axis does not have size one.
    numpy.exceptions.AxisError
        If an axis is out of range.

    See Also
    --------
    numpy.squeeze : Equivalent function for dense arrays.
    TTD.squeeze : Remove singleton dimensions from a TTD.
    expand_dims : Insert singleton dimensions into a TTD.
    get_item : Index into a TTD.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> ops.squeeze(TTD.ones((1, 2, 1))).shape
    (2,)

    """
    from wren_ttd.core import TTD

    if axis is None:
        axes = tuple(i for i, size in enumerate(ttd.shape) if size == 1)
    else:
        axes = normalize_axis_tuple(axis, ttd.ndim)
        if any(ttd.shape[i] != 1 for i in axes):
            raise ValueError(
                "cannot select an axis to squeeze out which has size not equal to one"
            )

    if not axes:
        return TTD(ttd.data.copy(), dtype=ttd.dtype)

    # use the indexing implementation for the value semantics for full squeeze
    indexes = tuple(0 if i in axes else slice(None) for i in range(ttd.ndim))
    return get_item(ttd, indexes)
