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
    """Remove singleton dimensions from a TTD."""
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
