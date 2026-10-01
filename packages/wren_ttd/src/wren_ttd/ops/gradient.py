# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from wren_ttd._numpy_api import implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("gradient")
def gradient[DType: np.floating](
    ttd: TTD[DType],
    *varargs: float | Sequence[float],
    axis: int | Sequence[int] | None = None,
    edge_order: Literal[1, 2] = 1,
) -> TTD[DType] | tuple[TTD[DType], ...]:
    """
    Compute the gradient of a TTD.

    Equivalent to :func:`numpy.gradient` for dense arrays, extended to
    TTDs. The gradient is computed using second order accurate central
    differences in the interior points and either first or second order
    accurate one-sided differences at the boundaries. Differentiation is
    applied to one core at a time while the other cores are copied
    unchanged, so each returned gradient has the same shape as the input.

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD to compute the gradient of. Contains samples of a scalar
        function on a (possibly non-uniform) grid.
    varargs : float | Sequence[float]
        Spacing between samples. Either a single scalar specifying the
        sample distance for the differentiated axes, one scalar per
        differentiated axis, or a single sequence of coordinates along
        the differentiated axis. The number of arguments must either match
        the number of differentiated axes or be a single argument applied
        to all of them. Defaults to unitary spacing.
    axis : int | Sequence[int] | None, optional
        The axis or axes along which to compute the gradient, by default
        None, which is equivalent to all axes. Entries may be negative,
        counting from the last axis, but must be unique.
    edge_order : Literal[1, 2], optional
        The order of the finite differences used at the boundaries, by
        default 1.

    Returns
    -------
    TTD[DType] | tuple[TTD[DType], ...]
        The derivatives of the TTD with respect to each differentiated
        axis, in the order of `axis`. A single TTD if only one axis was
        differentiated, otherwise a tuple of TTDs. Each derivative has
        the same shape as the input.

    See Also
    --------
    numpy.gradient : Equivalent function for dense arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.from_ndarray(np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0]))
    >>> np.asarray(ops.gradient(t))
    array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])

    """
    from wren_ttd.core import TTD

    if axis is None:
        axes = tuple(range(ttd.ndim))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)

    axes = normalize_axis_tuple(axes, ttd.ndim)

    if len(set(axes)) != len(axes):
        raise ValueError("axes entries must be unique")

    if len(varargs) == len(axes):
        step_args = [(v,) for v in varargs]
    elif len(varargs) == 1:
        step_args = [(varargs[0],)] * len(axes)
    elif not varargs:
        step_args = [()] * len(axes)
    else:
        raise ValueError("invalid number of arguments")

    results: list[TTD[DType]] = []
    for axis_idx, spacing in zip(axes, step_args, strict=True):
        # differentiate along only a single axis at a time and copy the rest unchanged
        cores = ttd.data.copy()

        # gradient of single axis returns an NDArray despite what typing suggests
        grad = np.gradient(cores[axis_idx], *spacing, axis=1, edge_order=edge_order)
        cores[axis_idx] = cast(Core[DType], cast(object, grad))

        results.append(TTD(cores, dtype=ttd.dtype))

    if len(results) == 1:
        return results[0]

    return tuple(results)
