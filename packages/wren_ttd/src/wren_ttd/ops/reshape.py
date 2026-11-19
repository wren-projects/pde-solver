# ruff: noqa: PLC0415
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from wren_ttd._helpers import merge_cores, split_core, to_int_tuple
from wren_ttd._numpy_api import DenseWarning, implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


@implements_function("reshape")
def reshape[DType: np.floating](
    a: TTD[DType],
    shape: int | Sequence[int],
    order: str = "C",
    *,
    copy: bool | None = None,  # noqa: ARG001
) -> TTD[DType]:
    """
    Reshape a TTD without changing its data.

    Equivalent to :func:`numpy.reshape` for dense arrays, extended to TTDs.
    The shape may be an int or a tuple of ints, with at most one entry set to
    ``-1`` to infer it from the total size.

    The operation is performed core by core. Adjacent modes are merged by
    contracting the shared TT rank, and a single mode is split by factoring
    the mode matrix with an SVD. Source and target shapes are aligned one mode
    at a time, so interleaved reshapes such as ``(2, 6) -> (3, 4)`` work.

    Parameters
    ----------
    a : TTD[DType]
        Input TTD tensor.
    shape : int or sequence of ints
        The target shape. At most one entry may be ``-1``.
    order : str, optional
        The order in which to read and write the elements. Only ``"C"`` is
        supported; ``"F"``, ``"A"``, and ``"K"`` raise
        :exc:`NotImplementedError`.
    copy : bool or None, optional
        Accepted for compatibility with :func:`numpy.reshape` and ignored: a
        new TTD is always returned.

    Returns
    -------
    TTD[DType]
        The reshaped TTD with the same dtype.

    Raises
    ------
    NotImplementedError
        If `order` is not ``"C"``, or if the target shape is empty or contains
        zero.
    ValueError
        If more than one ``-1`` is given, if a dimension is negative, or if
        the target shape has a different total size.

    See Also
    --------
    numpy.reshape : Equivalent function for dense arrays.
    ravel : Reshape a TTD to one dimension.
    flatten : Reshape a TTD to one dimension.

    Notes
    -----
    Merging adjacent modes is exact and preserves the TT ranks. Splitting a
    mode uses an untruncated SVD, so it is exact up to floating-point error but
    may inflate the ranks: the new rank is bounded by the smaller of the two
    flattened dimensions of the mode matrix. The result is always a valid TT
    representation.

    Keep in mind, that reshaping to shapes ``-1`` or ``a.size`` is equivalent to
    densifying the TTD and flattening the result into a single rank-1 core.

    Examples
    --------
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 3))
    >>> ops.reshape(t, (3, 2)).shape
    (3, 2)
    >>> ops.reshape(t, 6).shape
    (6,)
    >>> ops.reshape(t, (-1, 2)).shape
    (3, 2)

    """
    from wren_ttd.core import TTD

    if order != "C":
        raise NotImplementedError(
            f"reshape with order={order!r} is not supported for TTDs; "
            "only order='C' is supported"
        )

    target_shape = _resolve_shape(a.size, to_int_tuple(shape))

    if not target_shape:
        raise NotImplementedError("0-dimensional TTDs are not supported")

    if 0 in a.shape or 0 in target_shape:
        raise NotImplementedError("Reshaping zero-sized TTDs is not supported")

    if target_shape == a.shape:
        return TTD(a.data.copy(), dtype=a.dtype)

    if target_shape == (a.size,):
        warnings.warn(
            (
                "Calling `reshape` with an output shape of `a.size` "
                "effectively densifies the TTD. See Notes."
            ),
            DenseWarning,
            stacklevel=1,
        )
        dense_core = np.asarray(a).reshape(-1)[np.newaxis, :, np.newaxis]
        return TTD([dense_core], dtype=a.dtype)

    cores = _reshape_cores(a.data, target_shape)
    return TTD(cores, dtype=a.dtype)


def _resolve_shape(size: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve at most one ``-1`` entry and validate the total size."""
    unknown = [i for i, dimension in enumerate(shape) if dimension == -1]

    if len(unknown) > 1:
        raise ValueError("can only specify one unknown dimension")

    if any(dimension < -1 for dimension in shape):
        raise ValueError("negative dimensions are not allowed")

    if unknown:
        known_size = math.prod(dimension for dimension in shape if dimension != -1)
        if known_size == 0 or size % known_size != 0:
            raise ValueError(f"cannot reshape array of size {size} into shape {shape}")
        resolved = list(shape)
        resolved[unknown[0]] = size // known_size
        shape = tuple(resolved)

    if math.prod(shape) != size:
        raise ValueError(f"cannot reshape array of size {size} into shape {shape}")

    return shape


def _reshape_cores[DType: np.floating](
    cores: list[Core[DType]],
    target_shape: tuple[int, ...],
) -> list[Core[DType]]:
    """Regroup the modes of `cores` according to `target_shape`."""
    source_cores: list[Core[DType]] = list(reversed(cores))
    result: list[Core[DType]] = []

    for target in target_shape:
        core = source_cores.pop()
        mode = core.shape[1]

        # Accumulate adjacent modes until the group can be split at `target`.
        while mode < target or mode % target != 0:
            if not source_cores:
                raise ValueError(
                    "cannot reshape a TTD into the requested shape: "
                    "the total sizes do not match"
                )
            following = source_cores.pop()
            core = merge_cores(core, following)
            mode *= following.shape[1]

        if mode == target:
            result.append(core)
        else:
            head, tail = split_core(core, target)
            result.append(head)
            # the remainder starts the next target mode
            source_cores.append(tail)

    return result
