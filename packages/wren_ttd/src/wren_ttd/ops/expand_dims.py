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
    """Insert one or more singleton dimensions into a TTD."""
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
