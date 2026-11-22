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
    """Reverse one or more dimensions of a TTD."""
    from wren_ttd.core import TTD

    axes = normalize_axes(axis, ttd.ndim)
    cores = ttd.data.copy()
    for i in axes:
        cores[i] = cores[i][:, ::-1, :]

    return TTD(cores, dtype=ttd.dtype)
