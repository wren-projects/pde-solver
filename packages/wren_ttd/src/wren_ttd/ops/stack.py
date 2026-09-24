# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import islice
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.lib.array_utils import normalize_axis_index

from wren_ttd._helpers import block_core
from wren_ttd._numpy_api import implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("stack")
def stack[DType: np.floating](ttds: Sequence[TTD[DType]], axis: int = 0) -> TTD[DType]:
    """
    Stack TTDs along a new axis.

    Create a new TTD by stacking the given sequence of TTDs along the new axis.
    All TTDs must have the same shape and dtype.

    If only a TTD is given, treat its first axis as the sequence and stack the
    remaining axes.

    The resulting TTD will have significantly inflated ranks, so it is
    recommended to round it before performing further operations.

    Parameters
    ----------
    ttds : Sequence[TTD[DType]]
        The TTDs to stack.
    axis : int, optional
        The index of the new axis along which to stack the TTDs, by default 0.

    Returns
    -------
    TTD[DType]
        The stacked TTD.

    """
    from wren_ttd.core import TTD

    if isinstance(ttds, TTD) and ttds.ndim == 1:
        return cast(TTD[DType], ttds)

    # typing is dumb
    ttds = list(cast(Sequence[TTD[DType]], ttds))

    ttd0 = ttds[0]
    dtype = ttd0.dtype
    shape = ttd0.shape
    d = ttd0.ndim

    for ttd in ttds:
        if ttd.shape != shape:
            raise ValueError(f"Shape mismatch: {ttd.shape} != {shape}")

        if ttd.dtype != dtype:
            raise ValueError(f"Dtype mismatch: {ttd.dtype} != {dtype}")

    # normalize w.r.t. the new tensor
    axis = normalize_axis_index(axis, d + 1)

    # implement axis != 0 in terms of axis == 0
    if axis == d:
        return stack([ttd.T for ttd in ttds]).T
    if axis != 0:
        return stack(ttds).swapaxes(0, axis)

    M = len(ttds)

    # create the selector core for the new axis
    G0 = np.eye(M, dtype=dtype).reshape(1, M, M)

    # iterator of tuples of matching i-th cores from all given TTDs
    zipped_cores: Iterable[tuple[Core[DType], ...]] = zip(
        *(ttd.data for ttd in ttds), strict=True
    )

    cores: list[Core[DType]] = [
        G0,
        # stack all but the last cores into block cores
        *map(block_core, islice(zipped_cores, d - 1)),
        # stack last cores vertically
        np.vstack(next(zipped_cores)),
    ]

    return TTD(cores, dtype=dtype)
