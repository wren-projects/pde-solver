# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from wren_common.types import Scalar, ScalarTypes

from wren_ttd._numpy_api import implements_function
from wren_ttd.types import Core

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wren_ttd.core import TTD

type PadWidth = int | tuple[int, int] | Sequence[int] | Sequence[tuple[int, int]]

_PAD_PAIR_SIZE = 2

_UNEVEN_AXES_MESSAGE = (
    "TTD padding only supports uniform widths, but got pad_width={pad_width!r}; "
    "every axis must use the same (before, after) pair"
)
_UNEVEN_SIDES_MESSAGE = (
    "TTD padding only supports the same width on both sides of an axis, but got "
    "pad_width={pad_width!r}; use (width, width) instead of (before, after)"
)


def _padded_mask_core[DType: np.floating](
    n: int, pad_width: int, dtype: np.dtype[DType]
) -> Core[DType]:
    return np.ones((1, pad_width + n + pad_width, 1), dtype=dtype)


def _pad_constant[DType: np.floating](
    ttd: TTD[DType],
    pad_width: int,
    constant_value: Scalar,
) -> TTD[DType]:
    """Pad a TTD with constant values."""
    from wren_ttd.core import TTD

    # zero pad the TTD cores
    new_cores: list[Core[DType]] = []
    for core in ttd.data:
        r_left, n, r_right = core.shape
        new_core: Core[DType] = np.zeros(
            (r_left, pad_width + n + pad_width, r_right), dtype=ttd.dtype
        )
        new_core[:, pad_width : pad_width + n, :] = core
        new_cores.append(new_core)

    Z = TTD(new_cores, dtype=ttd.dtype)

    if constant_value == 0 or constant_value == ttd.dtype.type(0):
        return Z

    # Build an all-ones TTD of the padded shape (rank 1)
    ones_cores = (
        _padded_mask_core(orig_n, pad_width, dtype=ttd.dtype) for orig_n in ttd.shape
    )
    ones_ttd = TTD(ones_cores, dtype=ttd.dtype)

    # Build a rank-1 indicator TTD: 1 in the original domain, 0 in the padded region
    indicator_cores: list[Core[DType]] = []
    for orig_n in ttd.shape:
        ind_core: Core[DType] = _padded_mask_core(orig_n, pad_width, dtype=ttd.dtype)
        ind_core[0, :pad_width, 0] = 0
        ind_core[0, pad_width + orig_n :, 0] = 0
        indicator_cores.append(ind_core)
    indicator_ttd = TTD(indicator_cores, dtype=ttd.dtype)

    return Z + constant_value * (ones_ttd - indicator_ttd)


def _pad_width_int(value: object) -> int | None:
    """Return `value` as an integer, or None if it is not integer-like."""
    if isinstance(value, int):
        return int(value)
    if isinstance(value, np.integer):
        return value.item()
    return None


def _pad_width_pairs(pad_width: object, ndim: int) -> list[tuple[int, int]]:
    """
    Expand a NumPy-style `pad_width` into one (before, after) pair per axis.

    Raises
    ------
    NotImplementedError
        If `pad_width` cannot be expressed as one pair per axis.

    """
    width = _pad_width_int(pad_width)
    if width is not None:
        return [(width, width)] * ndim

    if not isinstance(pad_width, Iterable):
        raise NotImplementedError(
            "pad_width must be an integer or a sequence of (before, after) "
            f"pairs, got {pad_width!r}"
        )

    items: list[object] = list(pad_width)
    widths = [_pad_width_int(item) for item in items]
    if items and all(width is not None for width in widths):
        # NumPy's (before, after) form, broadcast to every axis.
        if len(widths) != _PAD_PAIR_SIZE:
            raise NotImplementedError(_UNEVEN_AXES_MESSAGE.format(pad_width=pad_width))
        before, after = cast("tuple[int, int]", tuple(widths))
        return [(before, after)] * ndim

    pairs: list[tuple[int, int]] = []
    for item in items:
        sides = list(item) if isinstance(item, Iterable) else None
        if sides is None or len(sides) != _PAD_PAIR_SIZE:
            raise NotImplementedError(_UNEVEN_AXES_MESSAGE.format(pad_width=pad_width))
        before = _pad_width_int(sides[0])
        after = _pad_width_int(sides[1])
        if before is None or after is None:
            raise NotImplementedError(_UNEVEN_AXES_MESSAGE.format(pad_width=pad_width))
        pairs.append((before, after))

    if len(pairs) != ndim:
        raise NotImplementedError(_UNEVEN_AXES_MESSAGE.format(pad_width=pad_width))

    return pairs


def _uniform_pad_width(pad_width: object, ndim: int) -> int:
    """
    Validate a NumPy-style `pad_width` and return its uniform width.

    Parameters
    ----------
    pad_width : object
        The padding specification to validate.
    ndim : int
        Number of axes of the padded tensor.

    Returns
    -------
    int
        The number of elements to pad on each side of every axis.

    Raises
    ------
    NotImplementedError
        If the padding is uneven across axes or sides, or if it is negative.

    """
    pairs = _pad_width_pairs(pad_width, ndim)
    before, after = pairs[0]

    if any(pair != (before, after) for pair in pairs):
        raise NotImplementedError(_UNEVEN_AXES_MESSAGE.format(pad_width=pad_width))
    if before != after:
        raise NotImplementedError(_UNEVEN_SIDES_MESSAGE.format(pad_width=pad_width))
    if before < 0:
        raise NotImplementedError(
            "TTD padding does not support negative widths (cropping); "
            f"got pad_width={pad_width!r}"
        )

    return before


@implements_function("pad")
def pad[DType: np.floating](
    array: TTD[DType],
    pad_width: PadWidth,
    mode: str = "constant",
    **kwargs: Any,  # type: ignore[reportAny]
) -> TTD[DType]:
    """
    Pad a TTD tensor.

    Parameters
    ----------
    array : TTD[DType]
        The TTD tensor to pad.
    pad_width : int or sequence of ints or sequence of pairs of ints
        The number of elements to pad on each side of each axis. Only uniform
        forms are supported: a scalar, ``(before, after)`` with equal sides, or
        one equal pair per axis such as ``((1, 1), (1, 1))``.
    mode : str, optional
        The padding mode. Only supported mode is the default 'constant' mode.
    **kwargs
        Additional keyword arguments for the padding mode.
        For 'constant' mode: constant_values (float, default 0.0). Only a
        scalar is supported.

    Returns
    -------
    TTD[DType]
        The padded TTD tensor.

    Raises
    ------
    NotImplementedError
        If `pad_width` is not uniform across axes and sides, if
        `constant_values` is not a scalar, if `mode` is not 'constant', or if
        any other keyword argument is given.

    See Also
    --------
    numpy.pad : Equivalent function for dense arrays.

    Examples
    --------
    >>> from wren_ttd.core import TTD
    >>> import numpy as np
    >>> t = TTD.ones((2, 3))
    >>> np.pad(t, 1).shape
    (4, 5)

    """
    width = _uniform_pad_width(pad_width, array.ndim)
    if width == 0:
        return array

    if mode != "constant":
        raise NotImplementedError(
            f"Pad mode {mode!r} is not supported; only 'constant' is implemented"
        )

    unsupported = set(kwargs) - {"constant_values"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise NotImplementedError(
            f"Unsupported keyword arguments for constant padding: {names}"
        )

    constant_values: Any = kwargs.get("constant_values", 0.0)  # type: ignore[reportAny]
    if isinstance(constant_values, (list, tuple, np.ndarray)):
        raise NotImplementedError(
            "Per-axis constant_values are not supported, use a scalar"
        )

    if isinstance(constant_values, ScalarTypes):
        constant_value = cast(Scalar, constant_values)
        return _pad_constant(array, width, constant_value)

    raise ValueError(
        f"Unsupported type for constant_values: {type(constant_values)}"  # type: ignore[reportAny]
    )
