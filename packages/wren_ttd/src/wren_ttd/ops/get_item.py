# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from wren_common.math import dot_product
from wren_common.types import Index1D, Matrix

from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def get_item[DType: np.floating](
    ttd: TTD[DType], indexes: Sequence[Index1D]
) -> TTD[DType] | DType:
    """
    Index into a TTD with basic NumPy-style indexing.

    Equivalent to indexing a :class:`numpy.ndarray` with integers and
    slices, e.g. ``ttd[0, 1, 2]`` or ``ttd[0, :, 2]``. Each entry of
    `indexes` addresses the corresponding core of the TTD: an integer
    index contracts away that mode, while a slice preserves it as a mode
    of the result. Cores past ``len(indexes)`` are kept unchanged.

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD to index into.
    indexes : Sequence[Index1D]
        One index per addressed mode, where each index is either an
        integer (or any :class:`SupportsIndex`) or a slice. Negative
        integer indices count from the end of the mode. Must not be empty
        and must not address more modes than the TTD has.

    Returns
    -------
    TTD[DType] | DType
        The indexed subtensor, or a scalar if every mode was indexed with
        an integer.

    See Also
    --------
    TTD.__getitem__ : Index into a TTD (also supports Ellipsis).

    Notes
    -----
    Only integer and slice indices are supported; boolean masks and
    advanced (fancy) indexing are not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 3))
    >>> np.asarray(ops.get_item(t, (0,)))
    array([1., 1., 1.])
    >>> ops.get_item(t, (0, 1))
    np.float64(1.0)

    """
    from wren_ttd.core import TTD

    if len(indexes) == 0:
        raise IndexError("Cannot index with an empty tuple")

    if len(indexes) > len(ttd.data):
        raise IndexError("Too many indices")

    message_matrix: Matrix[DType] = np.ones((1, 1), dtype=ttd.dtype)

    cores: list[Core[DType]] = []

    for core, index in zip(ttd.data, indexes, strict=False):
        if isinstance(index, int):
            message_matrix = cast(Matrix[DType], message_matrix @ core[:, index, :])
            continue

        # merge the slice of the core and the message matrix
        core_slice: Core[DType] = core[:, index, :]
        cores.append(dot_product(message_matrix, core_slice))
        # continue with a clean new message matrix
        message_matrix = np.eye(core.shape[2], dtype=ttd.dtype)

    remaining = ttd.data[len(indexes) :]

    # if all cores have been consumed, the result is a single element
    if not cores and not remaining:
        return ttd.dtype.type(message_matrix.squeeze())

    # merge the message matrix into either the first remaining core or the last
    # preserved core
    if remaining:
        remaining[0] = dot_product(message_matrix, remaining[0])
        cores.extend(remaining)
    else:
        cores[-1] = dot_product(cores[-1], message_matrix)

    return TTD(cores, dtype=ttd.dtype)
