# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from numpy.lib.array_utils import normalize_axis_index

from wren_ttd._helpers import block_core, normalize_dtype, promote_dtype
from wren_ttd._numpy_api import implements_function
from wren_ttd.ops.ravel import ravel
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("concatenate")
@implements_function("concat")
def concat[DType: np.floating](
    ttds: Sequence[TTD[DType]],
    axis: int | None = 0,
    out: object | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
    casting: str = "same_kind",
) -> TTD[DType]:
    """
    Join a sequence of TTDs along an existing axis.

    Equivalent to :func:`numpy.concatenate` for dense arrays, extended to
    TTDs. Also registered under ``concat``, the name of NumPy's
    :func:`numpy.concat` alias.

    The result is built exactly from the input cores. On every non-selected
    axis the cores are arranged block-diagonally; on the selected axis the
    input cores are placed in a zero-padded block layout. The first and last
    cores are concatenated along their shared rank, which is equivalent to
    the block layout because their outer ranks are one. The resulting
    TT-ranks are the sums of the input ranks, so concatenation inflates the
    ranks.

    Parameters
    ----------
    ttds : sequence of TTD[DType]
        The TTDs to concatenate. At least one is required. All inputs must
        have the same number of dimensions and matching shapes on every axis
        except `axis`.
    axis : int | None, optional
        The axis along which to concatenate. May be negative. If ``None``,
        every input is flattened first, by default 0.
    out : object, optional
        Not supported and must be ``None``. TTDs cannot be written by
        NumPy's dense concatenation machinery.
    dtype : np.dtype or type, optional
        The dtype of the result. If not provided, the input dtypes are
        promoted with NumPy's rules.
    casting : str, optional
        Only ``"same_kind"`` (the NumPy default) is supported.

    Returns
    -------
    TTD[DType]
        The concatenated TTD.

    Raises
    ------
    ValueError
        If `ttds` is empty, the input dimensions differ, or the shapes do
        not match on the non-concatenated axes.
    NotImplementedError
        If `out` is given, or if `casting` is not ``"same_kind"``.

    See Also
    --------
    numpy.concatenate : Equivalent function for dense arrays.
    numpy.concat : Alias of :func:`numpy.concatenate`.
    stack : Stack TTDs along a new axis.

    Notes
    -----
    ``axis=None`` first flattens every input with :func:`ravel` and then
    concatenates along axis 0. Since :func:`ravel` effectively densifies each
    input, a :class:`DenseWarning` is emitted in that case.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD.ones((2, 3))
    >>> b = TTD.full((1, 3), 2.0)
    >>> ops.concat([a, b]).shape
    (3, 3)
    >>> ops.concat([a, a], axis=-1).shape
    (2, 6)
    >>> ops.concat([a, b], axis=None).shape
    (9,)

    """
    from wren_ttd.core import TTD

    if out is not None:
        raise NotImplementedError("concat does not support out=")

    if casting != "same_kind":
        raise NotImplementedError(
            f"concat only supports casting='same_kind', not {casting!r}"
        )

    items = list(ttds)
    if not items:
        raise ValueError("need at least one TTD to concatenate")

    result_dtype = (
        normalize_dtype(dtype)
        if dtype is not None
        else promote_dtype(*(ttd.dtype for ttd in items))
    )
    items = [
        ttd if ttd.dtype == result_dtype else TTD(ttd, dtype=result_dtype)
        for ttd in items
    ]

    if axis is None:
        return concat([ravel(ttd) for ttd in items], axis=0, dtype=result_dtype)

    axis = _validate_shapes(items, axis)
    return TTD(_build_cores(items, axis, result_dtype), dtype=result_dtype)


def _validate_shapes[DType: np.floating](items: Sequence[TTD[DType]], axis: int) -> int:
    """Validate matching dimensions and non-concatenated shapes."""
    ndim = items[0].ndim
    if any(ttd.ndim != ndim for ttd in items):
        raise ValueError("all TTDs must have the same number of dimensions")

    axis = normalize_axis_index(axis, ndim)
    reference = items[0].shape
    for ttd in items:
        if any(ttd.shape[i] != reference[i] for i in range(ndim) if i != axis):
            raise ValueError(
                f"Shape mismatch: {ttd.shape} != {reference} on the axes "
                f"other than {axis}"
            )

    return axis


def _build_cores[DType: np.floating](
    items: Sequence[TTD[DType]], axis: int, dtype: np.dtype[DType]
) -> list[Core[DType]]:
    """Build the block-structured cores of the concatenated TTD."""
    ndim = items[0].ndim
    cores: list[Core[DType]] = []
    for i in range(ndim):
        axis_cores = [ttd.data[i] for ttd in items]
        if i == axis:
            cores.append(
                _axis_core(
                    axis_cores,
                    dtype,
                    shared_left=i == 0,
                    shared_right=i == ndim - 1,
                )
            )
        elif i == 0:
            # the left rank of every first core is one
            cores.append(cast(Core[DType], np.concatenate(axis_cores, axis=2)))
        elif i == ndim - 1:
            # the right rank of every last core is one
            cores.append(cast(Core[DType], np.concatenate(axis_cores, axis=0)))
        else:
            cores.append(block_core(tuple(axis_cores)))

    return cores


def _axis_core[DType: np.floating](
    cores: Sequence[Core[DType]],
    dtype: np.dtype[DType],
    *,
    shared_left: bool,
    shared_right: bool,
) -> Core[DType]:
    """Place cores in a zero-padded block layout along the mode axis."""
    left = 1 if shared_left else sum(core.shape[0] for core in cores)
    mode = sum(core.shape[1] for core in cores)
    right = 1 if shared_right else sum(core.shape[2] for core in cores)
    joined = np.zeros((left, mode, right), dtype=dtype)

    left_offset = mode_offset = right_offset = 0
    for core in cores:
        core_left, core_mode, core_right = core.shape
        joined[
            left_offset : left_offset + core_left,
            mode_offset : mode_offset + core_mode,
            right_offset : right_offset + core_right,
        ] = core
        if not shared_left:
            left_offset += core_left
        if not shared_right:
            right_offset += core_right
        mode_offset += core_mode

    return joined
