# ruff: noqa: PLC0415, PLR0913, FBT001, FBT002
# The API mirrors that of numpy.full_like, which fails those lints
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from wren_common.types import Scalar

from wren_ttd._numpy_api import implements_function

if TYPE_CHECKING:
    from wren_ttd.core import TTD


type ShapeLike = int | Sequence[int] | None

type _Order = Literal["C", "F", "A", "K"]


def _full_like[DType: np.floating](
    array: TTD[DType],
    fill_value: Scalar,
    dtype: np.dtype[DType] | None,
    order: _Order,
    subok: bool,
    shape: ShapeLike,
    device: str | None,
) -> TTD[DType] | np.ndarray:
    """Build a filled tensor with the shape and dtype of another tensor."""
    from wren_ttd.core import TTD

    if device not in (None, "cpu"):
        raise NotImplementedError("TTD operations are only supported on the CPU")

    shape = array.shape if shape is None else shape
    dtype = array.dtype if dtype is None else dtype

    if not subok:
        # np.full only accepts a concrete memory order, unlike the *_like
        # functions which also accept 'K' and 'A'.
        dense_order = "C" if order in {"K", "A"} else order
        return np.full(shape, fill_value, dtype=dtype, order=dense_order)

    return TTD.full(shape, fill_value, dtype=dtype)


@implements_function("full_like")
def full_like[DType: np.floating](
    array: TTD[DType],
    fill_value: Scalar,
    dtype: np.dtype[DType] | None = None,
    order: _Order = "K",
    subok: bool = True,
    shape: ShapeLike = None,
    *,
    device: str | None = None,
) -> TTD[DType] | np.ndarray:
    """
    Return a TTD filled with `fill_value`, with the shape and dtype of `array`.

    Equivalent to :func:`numpy.full_like` for dense arrays, extended to
    TTDs. The result is a rank-1 TTD with the shape and dtype of `array`,
    unless `shape` or `dtype` override them.

    Parameters
    ----------
    array : TTD[DType]
        The TTD whose shape and dtype are used as defaults for the result.
    fill_value : scalar
        Value used to fill the tensor.
    dtype : data-type, optional
        Overrides the data type of the result. Defaults to the dtype of
        `array`.
    order : {'C', 'F', 'A', 'K'}, optional
        Accepted for compatibility with :func:`numpy.full_like`. TTDs do
        not expose a memory order, so this does not affect the result.
    subok : bool, optional
        If True, return a TTD. If False, return a dense
        :class:`numpy.ndarray` instead.
    shape : int or sequence of ints, optional
        Overrides the shape of the result. Defaults to the shape of
        `array`.
    device : str, optional
        Accepted for compatibility with :func:`numpy.full_like`. Only the
        CPU is supported.

    Returns
    -------
    TTD[DType] or numpy.ndarray
        A tensor filled with `fill_value`.

    See Also
    --------
    numpy.full_like : Equivalent function for dense arrays.
    TTD.full : Return a new TTD filled with a constant value.
    ones_like : Return a TTD of ones with the shape and dtype of a TTD.
    zeros_like : Return a TTD of zeros with the shape and dtype of a TTD.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 2))
    >>> np.asarray(np.full_like(t, 3.0))
    array([[3., 3.],
           [3., 3.]])

    """
    return _full_like(array, fill_value, dtype, order, subok, shape, device)


@implements_function("ones_like")
def ones_like[DType: np.floating](
    array: TTD[DType],
    dtype: np.dtype[DType] | None = None,
    order: _Order = "K",
    subok: bool = True,
    shape: ShapeLike = None,
    *,
    device: str | None = None,
) -> TTD[DType] | np.ndarray:
    """
    Return a TTD of ones, with the shape and dtype of `array`.

    Equivalent to :func:`numpy.ones_like` for dense arrays, extended to
    TTDs. The result is a rank-1 TTD with the shape and dtype of `array`,
    unless `shape` or `dtype` override them.

    Parameters
    ----------
    array : TTD[DType]
        The TTD whose shape and dtype are used as defaults for the result.
    dtype : data-type, optional
        Overrides the data type of the result. Defaults to the dtype of
        `array`.
    order : {'C', 'F', 'A', 'K'}, optional
        Accepted for compatibility with :func:`numpy.ones_like`. TTDs do
        not expose a memory order, so this does not affect the result.
    subok : bool, optional
        If True, return a TTD. If False, return a dense
        :class:`numpy.ndarray` instead.
    shape : int or sequence of ints, optional
        Overrides the shape of the result. Defaults to the shape of
        `array`.
    device : str, optional
        Accepted for compatibility with :func:`numpy.ones_like`. Only the
        CPU is supported.

    Returns
    -------
    TTD[DType] or numpy.ndarray
        A tensor filled with ones.

    See Also
    --------
    numpy.ones_like : Equivalent function for dense arrays.
    TTD.ones : Return a new TTD of ones.
    full_like : Return a TTD filled with a constant value.
    zeros_like : Return a TTD of zeros with the shape and dtype of a TTD.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd.core import TTD
    >>> t = TTD.zeros((2, 2))
    >>> np.asarray(np.ones_like(t))
    array([[1., 1.],
           [1., 1.]])

    """
    return _full_like(array, 1, dtype, order, subok, shape, device)


@implements_function("zeros_like")
def zeros_like[DType: np.floating](
    array: TTD[DType],
    dtype: np.dtype[DType] | None = None,
    order: _Order = "K",
    subok: bool = True,
    shape: ShapeLike = None,
    *,
    device: str | None = None,
) -> TTD[DType] | np.ndarray:
    """
    Return a TTD of zeros, with the shape and dtype of `array`.

    Equivalent to :func:`numpy.zeros_like` for dense arrays, extended to
    TTDs. The result is a rank-1 TTD with the shape and dtype of `array`,
    unless `shape` or `dtype` override them.

    Parameters
    ----------
    array : TTD[DType]
        The TTD whose shape and dtype are used as defaults for the result.
    dtype : data-type, optional
        Overrides the data type of the result. Defaults to the dtype of
        `array`.
    order : {'C', 'F', 'A', 'K'}, optional
        Accepted for compatibility with :func:`numpy.zeros_like`. TTDs do
        not expose a memory order, so this does not affect the result.
    subok : bool, optional
        If True, return a TTD. If False, return a dense
        :class:`numpy.ndarray` instead.
    shape : int or sequence of ints, optional
        Overrides the shape of the result. Defaults to the shape of
        `array`.
    device : str, optional
        Accepted for compatibility with :func:`numpy.zeros_like`. Only the
        CPU is supported.

    Returns
    -------
    TTD[DType] or numpy.ndarray
        A tensor filled with zeros.

    See Also
    --------
    numpy.zeros_like : Equivalent function for dense arrays.
    TTD.zeros : Return a new TTD of zeros.
    full_like : Return a TTD filled with a constant value.
    ones_like : Return a TTD of ones with the shape and dtype of a TTD.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd.core import TTD
    >>> t = TTD.ones((2, 2))
    >>> np.asarray(np.zeros_like(t))
    array([[0., 0.],
           [0., 0.]])

    """
    return _full_like(array, 0, dtype, order, subok, shape, device)
