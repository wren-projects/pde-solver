from __future__ import annotations

from collections.abc import Callable, Iterable
from types import EllipsisType
from typing import Any, ParamSpec, Self, cast, final, overload, override

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin

from numpy_ttd.math import delta_truncated_svd, truncation_parameter
from numpy_ttd.types import Core, Matrix, NDArray

type AnyCallable = Callable[..., Any]

HANDLED_UFUNCS: dict[str, AnyCallable] = {}
HANDLED_FUNCTIONS: dict[str, AnyCallable] = {}

DEFAULT_EPSILON = np.float64(1e-6)


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


@final
class TTD[DType: np.floating](NDArrayOperatorsMixin):
    """
    Class for storing TTD encoded data.

    The class on the outside behaves like a NumPy NDArray but internally it
    stores the data in a compressed form using a TTD (Tensor Train
    decomposition). It also tries to perform all operations using this form but
    falls back on expanding to full NDArray if necessary.
    """

    data: list[Core[DType]]
    dtype: np.dtype[DType]

    __slots__ = ("data", "dtype")

    def __init__(
        self,
        data: Iterable[Core[DType]],
        *,
        dtype: np.dtype | None = None,
    ) -> None:
        """
        Create a new TTD object.

        Parameters
        ----------
        data : Iterable[Core[DType]]
            The cores of the TTD object. The caller must ensure that the data
            represents a valid TTD.
        dtype : np.dtype, optional
            The datatype of the TTD object. Defaults to the dtype of the first
            core. Either way, the dtype must be the same for all cores.

        Returns
        -------
            TTD object

        """
        super().__init__()

        if dtype is not None:
            self.dtype = dtype
            self.data = [core.astype(self.dtype) for core in data]
            if not self.data:
                raise ValueError("TTD must have at least one core")
            return

        self.data = data if isinstance(data, list) else list(data)
        if not self.data:
            raise ValueError("TTD must have at least one core")

        self.dtype = self.data[0].dtype
        if not all(core.dtype == self.dtype for core in self.data):
            raise ValueError("All cores must have the same dtype")

        self.dtype = self.data[0].dtype

        assert all(core.dtype == self.dtype for core in self.data)

    @staticmethod
    def from_ndarray[DT: np.floating](
        array: NDArray[DT], epsilon: np.floating | float = DEFAULT_EPSILON
    ) -> TTD[DT]:
        """
        Compress an NDArray into a TTD object.

        The resulting TTD satisfies ‖A - TTD‖ᶠ ≤ epsilon ⋅ ‖A‖ᶠ for given tensor
        A, where ‖A‖ᶠ is the Frobenius norm.

        Parameters
        ----------
        array : NDArray
            The tensor to compress.
        epsilon : float, optional
            The error tolerance for the compression. Uses system-wide default
            value if not provided.

        Returns
        -------
        TTD The TT-compressed object.

        """
        d = array.ndim

        if d == 1:
            return TTD([array.reshape((1, len(array), 1))])

        delta = truncation_parameter(array, epsilon)

        # 𝐂 = reshape(𝐂, [n, -1])
        residue: Matrix[DT] = array.reshape((array.shape[0], -1))

        # r₀ = 1
        r = 1

        cores: list[Core[DT]] = []

        # for k = 1 to d - 1 do
        # note: n = nₖ, r = rₖ₋₁
        for n in array.shape[:-1]:
            # 𝐂 = reshape(𝐂, [rₖ₋₁ nₖ, -1])
            residue = residue.reshape((r * n, -1))

            # 𝐔, 𝐒, 𝐕ᵀ = SVDᵟ(𝐂, δ)
            u, s, v_t = delta_truncated_svd(residue, delta)

            # 𝐆ₖ = reshape(U, [rₖ₋₁, nₖ, rₖ])
            new_core = u.reshape((r, n, -1))
            cores.append(new_core)

            # set rₖ = rankᵟ(C) for next iteration
            r = new_core.shape[2]

            # 𝐂 = 𝐒𝐕ᵀ
            # note: equivalent to np.diag(s) @ v_t
            residue = cast(Matrix[DT], np.einsum("i,ij->ij", s, v_t))

        cores.append(residue.reshape((*residue.shape, 1)))

        return TTD(cores)

    @override
    def __repr__(self) -> str:
        """Return a string representation of the TTD object."""
        return f"TTD({self.data})"

    @override
    def __str__(self) -> str:
        """Return a string representation of the TTD object."""
        return f"TTD({self.data})"

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the TTD object."""
        return tuple(core.shape[1] for core in self.data)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the TTD object."""
        return len(self.data)

    def __array__(
        self, dtype: npt.DTypeLike | None = None, *, copy: bool | None = None
    ) -> NDArray[DType]:
        """
        Expand a TTD object into a full NDArray.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        copy : bool, optional
            See :func:`numpy.asarray`. Supported only if the TTD represents a 1D
            vector.

        Returns
        -------
        NDArray[DType]
            The values in the series converted to a :class:`numpy.ndarray`
            with the specified `dtype`.

        """
        # Empty TTD
        if not self.data:
            return np.empty((0,), dtype=dtype)

        # 1D TTD
        if len(self.data) == 1:
            core = self.data[0]
            reshaped = core.reshape((core.shape[1],))
            return np.array(reshaped, dtype=dtype, copy=copy)

        if copy is False:
            raise ValueError(
                "`copy=False` is supported only for TTD representing a "
                "single-dimensional array."
            )

        # Multiply all cores together. This is equivalent to reduce(tensordot,
        # self.data), but faster since einsum does compute order optimizations
        # (at the cost of slightly uglier code).

        summation_indices: list[Any] = [
            item
            for i, core in enumerate(self.data)
            for item in (core, (2 * i, 2 * i + 1, 2 * i + 2))
        ]

        result = cast(
            NDArray[DType],
            np.einsum(*summation_indices, optimize=True),  # pyright: ignore[reportAny]
        )

        # remove singleton dimensions
        squeezed = result.squeeze()

        return squeezed if dtype is None else squeezed.astype(dtype)

    @override
    def __array_ufunc__(
        self,
        ufunc: Callable[ArrayUFuncParams, Any],
        method: str,
        *args: ArrayUFuncParams.args,
        **kwargs: ArrayUFuncParams.kwargs,
    ) -> TTD[DType] | NDArray[DType]:
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

        return cast(TTD[DType] | NDArray[DType], handler(*args, **kwargs))

    def __array_function__[*Args](
        self,
        func: Callable[[*Args], Any],
        types: tuple[type, ...],
        args: tuple[*Args],
        kwargs: dict[str, Any],
    ) -> TTD[DType] | NDArray[DType]:
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

        return cast(TTD[DType] | NDArray[DType], handler(*args, **kwargs))

    @implements_ufunc("sum")
    def sum(self: Self) -> float:
        """Sum the elements of the TTD object."""
        # NOTE: this is an example of ufunc, probably doesn't need to be implemented
        raise NotImplementedError

    # TODO: implement other NumPy functions

    def _to_raw(self) -> list[Core[DType]]:
        """Retrieve the internal representation as a list of NDArrays."""
        return self.data

    @overload
    def __getitem__(self, key: tuple[int, ...]) -> NDArray[DType]: ...
    @overload
    def __getitem__(self, key: EllipsisType) -> TTD[DType]: ...

    def __getitem__(
        self, key: EllipsisType | tuple[int, ...]
    ) -> TTD[DType] | NDArray[DType]:
        """Get a single values from the TTD object."""
        if key == Ellipsis:
            return self.__class__(a.copy() for a in self.data)

        raise NotImplementedError
