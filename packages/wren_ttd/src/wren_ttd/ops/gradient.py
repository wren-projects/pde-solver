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

    Parameters
    ----------
    ttd : TTD[DType]
        The TTD to compute the gradient of.
    varargs : float | Sequence[float]
        The step sizes to use for the gradient.
    axis : int | Sequence[SupportsIndex] | None, optional
        The axis or axes along which to compute the gradient, by default None,
        which is equivalent to all axes.
    edge_order : Literal[1, 2], optional
        The order of the finite differences used to compute the gradient, by
        default 1.

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
