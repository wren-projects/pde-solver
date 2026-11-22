# ruff: noqa: PLC0415, FBT001, FBT002
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from wren_ttd._helpers import singleton_cores, to_int_tuple
from wren_ttd._numpy_api import implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("broadcast_to")
def broadcast_to[DType: np.floating](
    ttd: TTD[DType],
    shape: int | Sequence[int],
    subok: bool = False,
) -> TTD[DType]:
    """
    Broadcast a TTD to a new shape.

    Equivalent to :func:`numpy.broadcast_to` for dense arrays, extended to
    TTDs. The target `shape` is aligned with ``ttd.shape`` from the right: each
    aligned source dimension must equal the corresponding target dimension or
    be ``1``. Extra target dimensions are added as new leading axes.

    Parameters
    ----------
    ttd : TTD[DType]
        Input TTD tensor.
    shape : int or sequence of ints
        The target shape.
    subok : bool, optional
        Not supported; must be ``False``.

    Returns
    -------
    TTD[DType]
        The broadcast TTD with the same dtype.

    Raises
    ------
    NotImplementedError
        If `subok` is ``True``.
    ValueError
        If the target shape is incompatible with ``ttd.shape``.

    See Also
    --------
    numpy.broadcast_to : Equivalent function for dense arrays.
    expand_dims : Insert singleton dimensions into a TTD.

    Notes
    -----
    New leading axes are added as identity cores, and a source axis of size
    ``1`` is expanded by repeating its mode slices, which is exact and keeps
    the TT ranks unchanged. No truncation is performed and the whole tensor is
    never densified.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((3,))
    >>> ops.broadcast_to(t, (2, 3)).shape
    (2, 3)
    >>> ops.broadcast_to(t, 3).shape
    (3,)

    """
    from wren_ttd.core import TTD

    if subok:
        raise NotImplementedError("`subok=True` is not supported for TTDs")

    target_shape = to_int_tuple(shape)

    if len(target_shape) < ttd.ndim:
        raise ValueError(
            "input operand has more dimensions than allowed by the axis remapping"
        )

    leading = len(target_shape) - ttd.ndim

    cores = singleton_cores(target_shape[:leading], ttd.dtype)

    for core, size in zip(ttd.data, target_shape[leading:], strict=True):
        mode = core.shape[1]

        if mode == size:
            cores.append(core)
        elif mode == 1:
            cores.append(cast(Core[DType], np.repeat(core, size, axis=1)))
        else:
            raise ValueError(
                "operands could not be broadcast together with remapped shapes "
                f"[original->remapped]: {ttd.shape} and requested shape "
                f"{target_shape}"
            )

    return TTD(cores, dtype=ttd.dtype)
