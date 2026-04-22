from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from types import EllipsisType
from typing import Any, ParamSpec, cast, final, overload, override

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin

from numpy_ttd import ops
from numpy_ttd._numpy_api import HANDLED_FUNCTIONS, HANDLED_UFUNCS
from numpy_ttd.math import delta_truncated_svd, qr_rows, truncation_parameter
from numpy_ttd.types import Core, Matrix, NDArray

DEFAULT_EPSILON = np.float64(1e-9)


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

        """
        super().__init__()

        if dtype is not None:
            self.dtype = dtype
            self.data = [core.astype(self.dtype) for core in data]
        else:
            self.data = list(data)
            self.dtype = self.data[0].dtype
            if not all(core.dtype == self.dtype for core in self.data):
                raise ValueError("All cores must have the same dtype")

        if not self.data:
            raise ValueError("TTD must have at least one core")

        if not all(
            a.shape[2] == b.shape[0]
            for a, b in zip(self.data, self.data[1:], strict=False)
        ):
            raise ValueError("Missmatch in core shapes")

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
        return str(self)

    @override
    def __str__(self) -> str:
        """Return a string representation of the TTD object."""
        return f"TTD(shape={self.shape},\n{'\n\n'.join(map(str, self.data))})"

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

        # Multiply all cores together. This is equivalent to a repeated
        # tensordot, but faster since einsum does compute order optimizations.
        # The generated expression looks like ABC,CDE,FGH,…, XYZ->ABCD…YZ, where
        # A and Z are singleton dimensions (from TTD) making the final
        # (squeezed) result BCD…Y.

        summation_indices: list[Any] = [
            item
            for i, core in enumerate(self.data)
            for item in (core, (2 * i, 2 * i + 1, 2 * i + 2))
        ]

        # einsum accepts besides a string, also an alternating list of indices
        # and tensors, e.g., (0,1), A, (2,3), B, (4,5), C, …

        result = cast(
            NDArray[DType],
            np.einsum(*summation_indices, optimize=True),  # pyright: ignore[reportAny]
        )

        squeezed = result.squeeze()

        return squeezed if dtype is None else squeezed.astype(dtype)

    def round(self, epsilon: DType | float = DEFAULT_EPSILON) -> None:
        """
        Round the TTD object by decreasing ranks.

        Uses SVD for compression. Ensures that the ranks of the rounded TTD 𝐀̃
        are maximally reduced, while ensuring that the relative error is less
        than `epsilon`.

        Operates on the TTD object in-place. See also :func:`TTD.rounded` to get
        a rounded copy.

        Parameters
        ----------
        epsilon : float, optional
            The relative error tolerance for the compression. Uses system-wide default
            value if not provided.

        """
        if self.ndim == 1:
            return

        # Suppose that 𝐀 is in the TT-format:
        # 𝐀(i₁, ..., i_d) = 𝐆₁(i₁) 𝐆₂(i₂) ... 𝐆_d(i_d)

        # (𝐆₁, ..., 𝐆_d)
        # Note: cores are 0 indexed here but 1 indexed in the paper, so 𝐆ₖ = G[k - 1]
        cores = self.data
        d = len(cores)

        for k in range(d, 1, -1):  # for k = d to 2 step -1
            # [𝐆ₖ(βₖ₋₁; iₖβₖ), R(αₖ₋₁, βₖ₋₁)] := QR_rows(𝐆ₖ(αₖ₋₁; iₖβₖ))
            # G = 𝐆ₖ(αₖ₋₁; iₖβₖ)
            core = cores[k - 1]
            alpha_k1, i_k, beta_k = core.shape
            # 𝐐, 𝐑 = QR_rows(𝐆ₖ(αₖ₋₁; iₖβₖ)) = QR(𝐆ₖ(αₖ₋₁; iₖβₖ)ᵀ)ᵀ
            q, r = qr_rows(core.reshape((alpha_k1, i_k * beta_k)))
            # 𝐆ₖ(βₖ₋₁; iₖβₖ) = 𝐐
            cores[k - 1] = q.reshape((-1, i_k, beta_k))
            # 𝐆ₖ₋₁ := 𝐆ₖ₋₁ ×₃ 𝐑
            # NOTE: there is a typo in the TTD paper: it incorrectly says 𝐆ₖ ×₃ 𝐑
            cores[k - 2] = np.einsum("ijk,kl", cores[k - 2], r)

        # this is necessary for Python's typing
        delta = truncation_parameter(cast(NDArray[DType], cast(Any, self)), epsilon)

        for k in range(1, d):  # for k = 1 to d-1
            # G = 𝐆ₖ(αₖ₋₁; iₖβₖ)
            core = cores[k - 1]
            beta_k1, i_k, beta_k = core.shape
            # 𝐔, 𝚲, 𝐕ᵀ := SVDᵟ(𝐆ₖ(βₖ₋₁; iₖβₖ))
            u, s, v_t = delta_truncated_svd(core.reshape(beta_k1 * i_k, beta_k), delta)
            # 𝐆ₖ(βₖ₋₁; iₖγₖ) = 𝐔
            cores[k - 1] = u.reshape((beta_k1, i_k, -1))
            # 𝐆ₖ₊₁ := 𝐆ₖ₊₁ ×₁ (𝐕𝚲)ᵀ
            cores[k] = np.einsum(
                "ijk,hi,h->hjk",
                cores[k],
                v_t,
                s,
                # in 99% of cases, this is the optimal path
                optimize=("einsum_path", (0, 1), (0, 1)),
            )

    def rounded(self, epsilon: DType | float = DEFAULT_EPSILON) -> TTD[DType]:
        """Return a new rounded TTD object."""
        ttd = self[...]
        ttd.round(epsilon)
        return ttd

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

        if not hasattr(ufunc, "__name__") or not isinstance(ufunc.__name__, str):
            # not a valid ufunc
            raise ValueError(f"Invalid ufunc: {ufunc}")

        handler = HANDLED_UFUNCS.get(ufunc.__name__)

        return (
            cast(TTD[DType] | NDArray[DType], handler(*args, **kwargs))
            if handler is not None
            else NotImplemented
        )

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
        if not hasattr(func, "__name__") or not isinstance(func.__name__, str):
            # not a valid function
            raise ValueError(f"Invalid array function: {func}")

        # Need to handle functions in submodules
        name = ".".join([*func.__module__.split(".")[1:], func.__name__])

        handler = HANDLED_FUNCTIONS.get(name)

        return (
            cast(TTD[DType] | NDArray[DType], handler(*args, **kwargs))
            if handler is not None
            else NotImplemented
        )

    def _get_item(self, indexes: tuple[int, ...]) -> NDArray[DType] | DType:
        """Retrieve a single value from the TTD object."""
        if len(indexes) == 0:
            raise IndexError("Cannot index with an empty tuple")

        if len(indexes) != len(self.data):
            raise IndexError(
                f"Cannot index with indexes {indexes}, "
                f"expected {len(self.data)} indexes"
            )

        result = functools.reduce(
            np.matmul,
            (core[:, j] for core, j in zip(self.data, indexes)),  # noqa: B905
        )

        return result.squeeze()

    @overload
    def __getitem__(self, key: tuple[int, ...]) -> NDArray[DType] | DType: ...
    @overload
    def __getitem__(self, key: EllipsisType) -> TTD[DType]: ...

    def __getitem__(
        self, key: EllipsisType | tuple[int, ...]
    ) -> TTD[DType] | NDArray[DType] | DType:
        """Get a single values from the TTD object."""
        if key == Ellipsis:
            return self.__class__(a.copy() for a in self.data)

        if isinstance(key, tuple):
            return self._get_item(key)

        raise NotImplementedError

    @override
    def __add__(self, other: TTD[DType]) -> TTD[DType]:
        """Add two TTD objects."""
        return ops.add(self, other)

    @override
    def __iadd__(self, other: TTD[DType]) -> TTD[DType]:
        """In-place add another tensor."""
        return ops.add(self, other, out=self)

    @override
    def __radd__(self, other: TTD[DType]) -> TTD[DType]:
        """Reverse add another tensor."""
        return ops.add(other, self)

    @override
    def __sub__(self, other: TTD[DType]) -> TTD[DType]:
        """Subtract two TTD objects."""
        return ops.add(self, -other)

    @override
    def __isub__(self, other: TTD[DType]) -> TTD[DType]:
        """In-place subtract another tensor."""
        return ops.add(self, -other, out=self)

    @override
    def __rsub__(self, other: TTD[DType]) -> TTD[DType]:
        """Reverse subtract another tensor."""
        return ops.add(-other, self)

    @override
    def __mul__(self, other: np.floating | float) -> TTD[DType]:
        """Multiply two TTD objects."""
        return ops.multiply(self, other)

    @override
    def __imul__(self, other: np.floating | float) -> TTD[DType]:
        """In-place multiply two TTD objects."""
        return ops.multiply(self, other, out=self)

    @override
    def __rmul__(self, other: np.floating | float) -> TTD[DType]:
        """Reverse multiply two TTD objects."""
        return ops.multiply(self, other)

    @override
    def __neg__(self) -> TTD[DType]:
        """Negate a TTD object."""
        return ops.neg(self)
