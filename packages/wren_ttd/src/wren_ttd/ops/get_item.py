# ruff: noqa: PLC0415
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, SupportsIndex, cast

import numpy as np
from wren_common.types import Index1D, Matrix, Scalar

from wren_ttd._helpers import mode_1_product, mode_3_product
from wren_ttd._numpy_api import DenseWarning
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


_BASIC_INDEX_TYPES = (SupportsIndex, slice)


def _is_advanced_index(key: tuple[object, ...]) -> bool:
    """Return whether a selection uses advanced (fancy) indexing."""
    return any(
        entry is not None
        and entry is not Ellipsis
        and (
            isinstance(entry, (np.ndarray, bool, np.bool_))
            or not isinstance(entry, _BASIC_INDEX_TYPES)
        )
        for entry in key
    )


def _normalize_basic_index(
    key: tuple[object, ...], ndim: int
) -> tuple[tuple[Index1D, ...], tuple[int, ...]]:
    """
    Expand a basic selection into a full tuple of per-mode indices.

    Ellipsis is expanded against the number of consumed modes, trailing modes
    are selected with ``slice(None)``, and ``None`` entries are removed. The
    positions of the removed ``None`` entries in the result are returned as
    insertion axes in output coordinates.
    """
    if sum(entry is Ellipsis for entry in key) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    consumed = sum(entry is not None and entry is not Ellipsis for entry in key)
    if consumed > ndim:
        raise IndexError(
            f"too many indices for array: array is {ndim}-dimensional, "
            f"but {consumed} were indexed"
        )

    expanded: list[object] = []
    for entry in key:
        if entry is Ellipsis:
            expanded.extend([slice(None)] * (ndim - consumed))
        else:
            expanded.append(entry)

    consumed = sum(entry is not None for entry in expanded)
    expanded.extend([slice(None)] * (ndim - consumed))

    selection: list[Index1D] = []
    insertion_axes: list[int] = []
    output_axis = 0
    for entry in expanded:
        if entry is None:
            insertion_axes.append(output_axis)
            output_axis += 1
            continue

        selection.append(cast(Index1D, entry))
        if not isinstance(entry, SupportsIndex):
            output_axis += 1

    return tuple(selection), tuple(insertion_axes)


def _advanced_get_item[DType: np.floating](
    ttd: TTD[DType], key: tuple[object, ...]
) -> TTD[DType] | DType:
    """Index a TTD through a dense fallback and recompress the result."""
    from wren_ttd.core import TTD

    result = cast(object, np.asarray(ttd)[cast(Any, key)])
    if not isinstance(result, np.ndarray):
        return ttd.dtype.type(cast(Scalar, result))
    if result.ndim == 0:
        return ttd.dtype.type(result[()])
    return TTD(result)


def _get_basic_item[DType: np.floating](
    ttd: TTD[DType], indexes: Sequence[Index1D]
) -> TTD[DType] | DType:
    """Index a TTD with integers and slices on a per-mode selection."""
    from wren_ttd.core import TTD

    # keep track of the message matrix, None == [1]
    message: Matrix[DType] | None = None

    cores: list[Core[DType]] = []

    for core, index in zip(ttd.data, indexes, strict=False):
        if isinstance(index, SupportsIndex):
            core_slice: Matrix[DType] = core[:, index, :]
            message = (
                core_slice
                if message is None
                else cast(Matrix[DType], message @ core_slice)
            )
            continue

        # merge the slice of the core and the message matrix
        sliced: Core[DType] = core[:, index, :]
        if message is None:
            # no message to apply, keep the slice unchanged
            cores.append(sliced)
        else:
            cores.append(mode_1_product(message, sliced))
        # continue with a clean new message matrix
        message = None

    remaining = ttd.data[len(indexes) :]

    # if all cores have been consumed, the result is a single element
    if not cores and not remaining:
        assert message is not None
        return ttd.dtype.type(message.squeeze())

    # merge the message matrix into either the first remaining core or the last
    # preserved core
    if remaining:
        if message is not None:
            remaining[0] = mode_1_product(message, remaining[0])

        cores.extend(remaining)
    elif message is not None:
        cores[-1] = mode_3_product(cores[-1], message)

    return TTD(cores, dtype=ttd.dtype)


def get_item[DType: np.floating](ttd: TTD[DType], key: object) -> TTD[DType] | DType:
    """
    Index into a TTD with NumPy-style indexing.

    Equivalent to indexing a :class:`numpy.ndarray`, e.g. ``ttd[0, 1, 2]``,
    ``ttd[0, :, 2]``, ``ttd[None]``, or ``ttd[ttd > 0]``. Integers contract
    away a mode, slices preserve it, ``None`` inserts a singleton mode, and
    ``...`` selects all remaining modes. Trailing modes are selected
    implicitly. All-integer selections return a NumPy scalar; otherwise a TTD
    is returned.

    Basic indexing with integers, slices, ``None``, and ``...`` is exact.
    Advanced indexing with integer arrays, boolean masks, or other array-like
    indices falls back on a dense array, emits a :class:`DenseWarning`, and
    recompresses the result.

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD to index into.
    key : object
        A NumPy-style index: an integer, slice, ``None``, ``...``, an array,
        or a tuple of these.

    Returns
    -------
    TTD[DType] | DType
        The indexed subtensor, or a scalar if every mode was indexed with an
        integer and no singleton modes were inserted.

    Raises
    ------
    IndexError
        If the index is invalid, contains more than one ellipsis, or has too
        many entries for the tensor.

    See Also
    --------
    TTD.__getitem__ : Index into a TTD.

    Notes
    -----
    Advanced indexing densifies the tensor, since the sparse TT structure that
    survives an arbitrary fancy index is not known in general.

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
    >>> ops.get_item(t, None).shape
    (1, 2, 3)

    """
    from wren_ttd.core import TTD
    from wren_ttd.ops.expand_dims import expand_dims

    normalized_key = cast(tuple[object, ...], key) if isinstance(key, tuple) else (key,)

    if _is_advanced_index(normalized_key):
        warnings.warn(
            "Advanced indexing densifies the TTD; the result is recompressed",
            DenseWarning,
            stacklevel=3,
        )
        return _advanced_get_item(ttd, normalized_key)

    selection, insertion_axes = _normalize_basic_index(normalized_key, ttd.ndim)
    result = _get_basic_item(ttd, selection)

    if not insertion_axes:
        return result

    if not isinstance(result, TTD):
        return TTD.full((1,) * len(insertion_axes), result, dtype=ttd.dtype)

    return expand_dims(result, insertion_axes)
