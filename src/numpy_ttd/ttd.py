from collections.abc import Callable, Iterable
from types import EllipsisType
from typing import Any, ParamSpec, Self, cast, overload, override

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin

from numpy_ttd.types import Core, NDArray

type AnyCallable = Callable[..., Any]

HANDLED_UFUNCS: dict[str, AnyCallable] = {}
HANDLED_FUNCTIONS: dict[str, AnyCallable] = {}


def implements_ufunc[F: AnyCallable](name: str) -> Callable[[F], F]:
    """Register an `__array_ufunc__` implementation for TTD objects."""

    def decorator(func: F) -> F:
        HANDLED_UFUNCS[name] = func
        return func

    return decorator


def implements_function[F: AnyCallable](name: str) -> Callable[[F], F]:
    """Register an `__array_function__` implementation for TTD objects."""

    def decorator(func: F) -> F:
        HANDLED_FUNCTIONS[name] = func
        return func

    return decorator


ArrayFunctionParams = ParamSpec("ArrayFunctionParams")
ArrayUFuncParams = ParamSpec("ArrayUFuncParams")


class TTD[DType: np.floating](NDArrayOperatorsMixin):
    """
    Class for storing TTD encoded data.

    The class on the outside behaves like a NumPy NDArray but internally it
    stores the data in a compressed form using a TTD (tensor train
    decomposition). It also tries to perform all operations using this form but
    falls back on expanding to full NDArray if necessary.
    """

    cores: list[Core[DType]]

    def __init__(self, data: Iterable[Core[DType]] | None = None) -> None:
        """
        Create a new empty TTD object.

        Parameters
        ----------
        data : Iterable[NDArray], optional
            The data to store in the TTD object. If provided, the caller must
            ensure that the data represents a valid TTD. Defaults to an empty
            decomposition.

        Returns
        -------
            TTD object

        """
        super().__init__()
        self.cores = list(data) if data is not None else []

    @staticmethod
    def from_ndarray[DT: np.floating](_array: NDArray[DT]) -> "TTD[DT]":
        """Compress an NDArray into a TTD object."""
        # TODO: implement compression
        raise NotImplementedError

    @override
    def __repr__(self) -> str:
        """Return a string representation of the TTD object."""
        return f"TTD({self.cores})"

    @override
    def __str__(self) -> str:
        """Return a string representation of the TTD object."""
        return f"TTD({self.cores})"

    def __array__(
        self, dtype: npt.DTypeLike | None = None, *, copy: bool | None = None
    ) -> NDArray[DType]:
        """
        Expand a TTD object into a full NDArray.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        copy : bool or None, optional
            See :func:`numpy.asarray`.

        Returns
        -------
        NDArray
            The values in the series converted to a :class:`numpy.ndarray`
            with the specified `dtype`.

        """
        raise NotImplementedError

    @override
    def __array_ufunc__(
        self,
        ufunc: Callable[ArrayUFuncParams, Any],
        method: str,
        *args: ArrayUFuncParams.args,
        **kwargs: ArrayUFuncParams.kwargs,
    ) -> "TTD[DType]" | NDArray[DType]:
        """
        Apply a NumPy ufunc to a TTD object.

        Parameters
        ----------
        ufunc : Callable
            The NumPy ufunc to apply.
        method : str
            The method to use for the ufunc.
        *args : list
            The inputs to the ufunc.
        **kwargs : dict
            The keyword arguments to the ufunc.

        Returns
        -------
        TTD | NDArray
            The result of the ufunc applied to the TTD object.

        """
        if method != "__call__":
            # only handle callable ufuncs
            return NotImplemented

        handler = HANDLED_UFUNCS.get(ufunc.__name__)

        if handler is None:
            return NotImplemented

        return cast("TTD[DType]" | NDArray[DType], handler(*args, **kwargs))

    def __array_function__[*Args](
        self,
        func: Callable[[*Args], Any],
        types: tuple[type, ...],
        args: tuple[*Args],
        kwargs: dict[str, Any],
    ) -> "TTD[DType]" | NDArray[DType]:
        """
        Call a NumPy method on a TTD object.

        Parameters
        ----------
        func : Callable
            The NumPy method to call.
        types : tuple[type]
            The types of the arguments.
        args : tuple
            The arguments to the numpy method.
        kwargs : dict
            The keyword arguments to the numpy method.

        Returns
        -------
        TTD | NDArray
            The result of the NumPy method applied to the TTD object.

        """
        # Need to handle functions in submodules
        name = ".".join([*func.__module__.split(".")[1:], func.__name__])

        handler = HANDLED_FUNCTIONS.get(name)

        if handler is None:
            return NotImplemented

        return cast("TTD[DType]" | NDArray[DType], handler(*args, **kwargs))

    @implements_ufunc("sum")
    def sum(self: Self) -> float:
        """Sum the elements of the TTD object."""
        # NOTE: this is an example of ufunc, probably doesn't need to be implemented
        raise NotImplementedError

    # TODO: implement other NumPy functions

    def _to_raw(self) -> list[Core[DType]]:
        """Retrieve the internal representation as a list of NDArrays."""
        return self.cores

    @overload
    def __getitem__(self, key: tuple[int, ...]) -> NDArray[DType]: ...
    @overload
    def __getitem__(self, key: EllipsisType) -> "TTD[DType]": ...

    def __getitem__(
        self, key: EllipsisType | tuple[int, ...]
    ) -> "TTD[DType]" | NDArray[DType]:
        """Get a single values from the TTD object."""
        if key == Ellipsis:
            return self.__class__(a.copy() for a in self.cores)

        raise NotImplementedError
