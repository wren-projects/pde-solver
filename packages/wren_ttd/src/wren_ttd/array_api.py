# pyright: reportAny=false
"""
Array API namespace for TTD arrays.

The module implements the subset of the Python Array API standard supported by
:class:`wren_ttd.core.TTD`. Constructors return TTDs, registered operations are
bound directly to their TTD implementations, and every other attribute is
delegated to NumPy, whose functions dispatch on ``__array_function__`` and
``__array_ufunc__``. Nested namespaces such as :data:`linalg` bind their
registered operations the same way. This keeps the namespace small while
reusing the existing TTD registrations.

TTD tensors must have at least one dimension. Arrays of rank zero are not
representable, so constructors and :func:`asarray` raise
:class:`NotImplementedError` instead of silently returning a dense scalar.

Examples
--------
>>> import numpy as np
>>> from wren_ttd import array_api as xp
>>> x = xp.asarray([[1.0, 2.0], [3.0, 4.0]])
>>> np.asarray(xp.sum(x, axis=0))
array([4., 6.])

"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any, cast, override

import numpy as np
import numpy.typing as npt
from wren_common.types import Scalar

from wren_ttd import ops
from wren_ttd._helpers import normalize_dtype, to_int_tuple
from wren_ttd._numpy_api import get_operation, get_operations
from wren_ttd.core import TTD
from wren_ttd.types import TTDType

__array_api_version__ = np.__array_api_version__


def __getattr__(name: str) -> Any:
    """
    Return a registered TTD operation or delegate the attribute to NumPy.

    Parameters
    ----------
    name : str
        The requested attribute name.

    Returns
    -------
    Any
        The registered TTD operation, or the corresponding NumPy attribute.

    Raises
    ------
    AttributeError
        If `name` starts with an underscore or no implementation exists.

    """
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    operation = get_operation(name)
    if operation is not None:
        return operation
    return getattr(np, name)


class _Namespace(ModuleType):
    """Nested namespace with direct operation bindings and a NumPy fallback."""

    _prefix: str
    _fallback: Any

    def __init__(self, name: str, prefix: str, fallback: Any) -> None:
        super().__init__(name)
        self._prefix = prefix
        self._fallback = fallback

    @override
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                f"namespace {self._prefix!r} has no attribute {name!r}"
            )
        operation = get_operation(f"{self._prefix}.{name}")
        if operation is not None:
            return operation
        return getattr(self._fallback, name)

    @override
    def __dir__(self) -> list[str]:
        return sorted({*get_operations(self._prefix), *dir(self._fallback)})


linalg = _Namespace(f"{__name__}.linalg", "linalg", np.linalg)


def _validate_device(device: str | None) -> None:
    """Reject devices other than the CPU."""
    if device not in (None, "cpu"):
        raise ValueError("TTD arrays are only supported on the CPU")


def _validate_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    """Normalize a shape and reject rank-zero tensors."""
    result = to_int_tuple(shape)
    if not result:
        raise NotImplementedError("TTD arrays must have at least one dimension")
    return result


def _from_array(array: np.ndarray[Any, Any]) -> TTD[Any]:
    """Compress a dense array into a TTD, rejecting rank-zero input."""
    if array.ndim == 0:
        raise NotImplementedError("TTD arrays must have at least one dimension")
    return TTD(array)


def asarray(
    obj: object,
    /,
    *,
    dtype: npt.DTypeLike | None = None,
    device: str | None = None,
    copy: bool | None = None,
) -> TTD[Any]:
    """
    Convert the input to a TTD.

    Parameters
    ----------
    obj : TTD | array-like
        The input to convert. Scalar input is rejected because TTD has no
        rank-zero representation.
    dtype : dtype, optional
        The desired data type. TTD supports floating-point dtypes only.
    device : str, optional
        The device where the array is placed. Only ``"cpu"`` is supported.
    copy : bool, optional
        If ``True``, always copy the input. If ``False``, never copy and raise
        :class:`ValueError` when a copy would be required. If ``None``, copy
        only when necessary.

    Returns
    -------
    TTD
        The converted array.

    Raises
    ------
    NotImplementedError
        If the input has rank zero.
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `device` is not the CPU or if `copy` is ``False`` but a copy is
        required.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> np.asarray(xp.asarray([[1.0, 2.0]]))
    array([[1., 2.]])

    """
    _validate_device(device)

    if isinstance(obj, TTD):
        ttd = cast("TTD[Any]", obj)
        converted = ttd if dtype is None else ops.astype(ttd, dtype, copy=False)
        if copy is False:
            if converted is not ttd:
                raise ValueError("Unable to avoid copy in TTD.asarray")
            return ttd
        if copy is True:
            return ops.copy(converted)
        return converted

    if copy is False:
        raise ValueError("Unable to avoid copy in TTD.asarray")

    array = np.asarray(obj, dtype=dtype)
    return _from_array(array)


def zeros(
    shape: int | Sequence[int],
    /,
    *,
    dtype: TTDType[Any] | None = None,
    device: str | None = None,
) -> TTD[Any]:
    """
    Return a new TTD filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new tensor. Rank zero is not supported.
    dtype : dtype, optional
        The desired data type. Defaults to ``numpy.float64``.
    device : str, optional
        The device where the array is placed. Only ``"cpu"`` is supported.

    Returns
    -------
    TTD
        Tensor of zeros with the given shape and dtype.

    Raises
    ------
    NotImplementedError
        If `shape` has rank zero.
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `device` is not the CPU.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> np.asarray(xp.zeros((2, 2)))
    array([[0., 0.],
           [0., 0.]])

    """
    _validate_device(device)
    return TTD.zeros(_validate_shape(shape), dtype=dtype)


def ones(
    shape: int | Sequence[int],
    /,
    *,
    dtype: TTDType[Any] | None = None,
    device: str | None = None,
) -> TTD[Any]:
    """
    Return a new TTD filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new tensor. Rank zero is not supported.
    dtype : dtype, optional
        The desired data type. Defaults to ``numpy.float64``.
    device : str, optional
        The device where the array is placed. Only ``"cpu"`` is supported.

    Returns
    -------
    TTD
        Tensor of ones with the given shape and dtype.

    Raises
    ------
    NotImplementedError
        If `shape` has rank zero.
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `device` is not the CPU.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> np.asarray(xp.ones((2, 2)))
    array([[1., 1.],
           [1., 1.]])

    """
    _validate_device(device)
    return TTD.ones(_validate_shape(shape), dtype=dtype)


def empty(
    shape: int | Sequence[int],
    /,
    *,
    dtype: TTDType[Any] | None = None,
    device: str | None = None,
) -> TTD[Any]:
    """
    Return a new TTD without initializing its values.

    A compressed tensor has no uninitialized memory, so this function returns
    zeros. The values must not be relied upon, matching the Array API contract.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new tensor. Rank zero is not supported.
    dtype : dtype, optional
        The desired data type. Defaults to ``numpy.float64``.
    device : str, optional
        The device where the array is placed. Only ``"cpu"`` is supported.

    Returns
    -------
    TTD
        Tensor with the given shape and dtype.

    Raises
    ------
    NotImplementedError
        If `shape` has rank zero.
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `device` is not the CPU.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> np.asarray(xp.empty(2)).shape
    (2,)

    """
    return zeros(shape, dtype=dtype, device=device)


def full(
    shape: int | Sequence[int],
    fill_value: Scalar,
    /,
    *,
    dtype: TTDType[Any] | None = None,
    device: str | None = None,
) -> TTD[Any]:
    """
    Return a new TTD filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new tensor. Rank zero is not supported.
    fill_value : scalar
        Value used to fill the tensor.
    dtype : dtype, optional
        The desired data type. Defaults to ``numpy.float64``.
    device : str, optional
        The device where the array is placed. Only ``"cpu"`` is supported.

    Returns
    -------
    TTD
        Tensor filled with `fill_value`, with the given shape and dtype.

    Raises
    ------
    NotImplementedError
        If `shape` has rank zero.
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `device` is not the CPU.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> np.asarray(xp.full((2, 2), 3.5))
    array([[3.5, 3.5],
           [3.5, 3.5]])

    """
    _validate_device(device)
    return TTD.full(_validate_shape(shape), fill_value, dtype=dtype)


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: TTDType[Any] | None = None,
    device: str | None = None,
) -> TTD[Any]:
    """
    Return evenly spaced values in a TTD.

    Parameters
    ----------
    start : int or float
        Start of the interval, or the stop when `stop` is omitted.
    stop : int or float, optional
        End of the interval.
    step : int or float, optional
        Spacing between values. Defaults to 1.
    dtype : dtype, optional
        The desired data type. Defaults to ``numpy.float64``.
    device : str, optional
        The device where the array is placed. Only ``"cpu"`` is supported.

    Returns
    -------
    TTD
        One-dimensional TTD with evenly spaced values.

    Raises
    ------
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `device` is not the CPU.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> np.asarray(xp.arange(3))
    array([0., 1., 2.])

    """
    _validate_device(device)
    resolved = None if dtype is None else normalize_dtype(dtype)
    array = (
        np.arange(start, stop, step, dtype=resolved)
        if stop is not None
        else np.arange(start, step=step, dtype=resolved)
    )
    return _from_array(array)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: TTDType[Any] | None = None,
    device: str | None = None,
) -> TTD[Any]:
    """
    Return a two-dimensional TTD with ones on the `k`-th diagonal.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    n_cols : int, optional
        Number of columns. Defaults to `n_rows`.
    k : int, optional
        Index of the diagonal. Defaults to the main diagonal.
    dtype : dtype, optional
        The desired data type. Defaults to ``numpy.float64``.
    device : str, optional
        The device where the array is placed. Only ``"cpu"`` is supported.

    Returns
    -------
    TTD
        Two-dimensional identity-like TTD.

    Raises
    ------
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `device` is not the CPU.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> np.asarray(xp.eye(2))
    array([[1., 0.],
           [0., 1.]])

    """
    _validate_device(device)
    resolved = None if dtype is None else normalize_dtype(dtype)
    return _from_array(np.eye(n_rows, n_cols, k=k, dtype=resolved))


def linspace(  # noqa: PLR0913
    start: int | float,
    stop: int | float,
    /,
    num: int,
    *,
    dtype: TTDType[Any] | None = None,
    device: str | None = None,
    endpoint: bool = True,
) -> TTD[Any]:
    """
    Return evenly spaced numbers over a closed interval as a TTD.

    Parameters
    ----------
    start : int or float
        Start of the interval.
    stop : int or float
        End of the interval.
    num : int
        Number of samples.
    dtype : dtype, optional
        The desired data type. Defaults to ``numpy.float64``.
    device : str, optional
        The device where the array is placed. Only ``"cpu"`` is supported.
    endpoint : bool, optional
        Whether `stop` is included. Defaults to ``True``.

    Returns
    -------
    TTD
        One-dimensional TTD with evenly spaced values.

    Raises
    ------
    TypeError
        If `dtype` is not a floating-point dtype.
    ValueError
        If `device` is not the CPU.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> np.asarray(xp.linspace(0, 1, 3))
    array([0. , 0.5, 1. ])

    """
    _validate_device(device)
    resolved = None if dtype is None else normalize_dtype(dtype)
    return _from_array(np.linspace(start, stop, num, dtype=resolved, endpoint=endpoint))


def _as_dtype(value: object) -> np.dtype[Any]:
    """Return the dtype of a TTD and convert any other object to a dtype."""
    if isinstance(value, TTD):
        ttd = cast("TTD[Any]", value)
        return np.dtype(ttd.dtype)
    return np.dtype(cast(npt.DTypeLike, value))


def result_type(*arrays_and_dtypes: object) -> np.dtype[Any]:
    """
    Return the dtype that results from type promotion of the arguments.

    TTD arguments contribute their dtype; every other argument is passed to
    :func:`numpy.result_type` unchanged.

    Parameters
    ----------
    *arrays_and_dtypes : object
        The operands to promote. TTD operands contribute their dtype; NumPy
        arrays, dtypes, and scalars are passed to :func:`numpy.result_type`.

    Returns
    -------
    numpy.dtype
        The promoted dtype.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> xp.result_type(xp.asarray([1.0], dtype=np.float32), np.float64)
    dtype('float64')

    """
    return np.result_type(*(_as_dtype(argument) for argument in arrays_and_dtypes))


def can_cast(from_: object, to: object, /) -> bool:
    """
    Return whether a cast from `from_` to `to` is allowed.

    Parameters
    ----------
    from_ : dtype | TTD
        Source dtype, or an array whose dtype is used.
    to : dtype | TTD
        Target dtype, or an array whose dtype is used.

    Returns
    -------
    bool
        ``True`` if the cast is allowed under NumPy's casting rules.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> xp.can_cast(np.float32, np.float64)
    True

    """
    return np.can_cast(_as_dtype(from_), _as_dtype(to))


def finfo(type: object, /) -> np.finfo[Any]:  # noqa: A002
    """
    Return machine limits for floating-point types.

    Parameters
    ----------
    type : dtype | TTD
        The dtype whose limits are returned.

    Returns
    -------
    numpy.finfo
        Machine limits for the requested floating-point type.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> xp.finfo(np.float32).bits
    32

    """
    return np.finfo(_as_dtype(type))


def iinfo(type: object, /) -> np.iinfo[Any]:  # noqa: A002
    """
    Return machine limits for integer types.

    Parameters
    ----------
    type : dtype | TTD
        The dtype whose limits are returned.

    Returns
    -------
    numpy.iinfo
        Machine limits for the requested integer type.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> xp.iinfo(np.int8).max
    127

    """
    return np.iinfo(_as_dtype(type))


def isdtype(dtype: object, kind: object) -> bool:
    """
    Return whether a dtype belongs to the requested kind.

    Parameters
    ----------
    dtype : dtype | TTD
        The dtype to test.
    kind : str or tuple
        A dtype kind name, a dtype, or a tuple of either.

    Returns
    -------
    bool
        ``True`` if the dtype belongs to `kind`.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import array_api as xp
    >>> xp.isdtype(np.float32, "real floating")
    True
    >>> xp.isdtype(np.int32, "real floating")
    False

    """
    return np.isdtype(_as_dtype(dtype), kind)  # type: ignore[arg-type]
