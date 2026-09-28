from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from types import EllipsisType
from typing import Any, ParamSpec, SupportsIndex, cast, final, overload, override

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin
from wren_common.math import dot_product, scale_matrix
from wren_common.types import Index1D, Matrix, NDArray, Scalar

from wren_ttd import ops
from wren_ttd._helpers import (
    orthogonalize_right,
    reverse_cores,
    to_int_tuple,
    truncation_parameter,
)
from wren_ttd._numpy_api import HANDLED_FUNCTIONS, HANDLED_UFUNCS, implements_function
from wren_ttd.math import DEFAULT_EPSILON, delta_truncated_svd
from wren_ttd.types import Core

ArrayFunctionParams = ParamSpec("ArrayFunctionParams")
ArrayUFuncParams = ParamSpec("ArrayUFuncParams")


@final
class TTD[DType: np.floating](NDArrayOperatorsMixin, Sequence["TTD[DType]" | DType]):
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

        for a, b in zip(self.data, self.data[1:], strict=False):
            if a.shape[2] != b.shape[0]:
                raise ValueError(
                    f"Missmatch in core shapes: {a.shape} does not match {b.shape}"
                )

        if not (self.data[0].shape[0] == self.data[-1].shape[2] == 1):
            raise ValueError("The boundary ranks have to be 1")

    @staticmethod
    def from_ndarray[DT: np.floating](
        array: NDArray[DT], epsilon: Scalar = DEFAULT_EPSILON
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
        TTD
            The compressed TTD.

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
            residue = scale_matrix(s, v_t)

        cores.append(residue.reshape((*residue.shape, 1)))

        return TTD(cores)

    @staticmethod
    def ones[DT: np.floating](
        shape: int | Sequence[int],
        *,
        dtype: np.dtype[DT] | None = None,
    ) -> TTD[DT]:
        """
        Return a new TTD representing a tensor of ones.

        Equivalent to :func:`numpy.ones`, but returns the tensor in
        compressed TTD format with all TT-ranks equal to 1.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new tensor, e.g. ``(2, 3)`` or ``2``.
        dtype : data-type, optional
            The desired data-type of the tensor, e.g. ``numpy.float64``.
            The default is ``numpy.float64``.

        Returns
        -------
        TTD
            Tensor of ones with the given shape and dtype.

        See Also
        --------
        numpy.ones : Equivalent function for dense arrays.
        TTD.zeros : Return a new TTD of zeros.
        TTD.full : Return a new TTD filled with a constant value.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 2))
        >>> np.asarray(t)
        array([[1., 1.],
               [1., 1.]])

        """
        cores = [np.ones((1, n, 1), dtype=dtype) for n in to_int_tuple(shape)]
        return TTD(cores, dtype=dtype)

    @staticmethod
    def zeros[DT: np.floating](
        shape: int | Sequence[int],
        *,
        dtype: np.dtype[DT] | None = None,
    ) -> TTD[DT]:
        """
        Return a new TTD representing a tensor of zeros.

        Equivalent to :func:`numpy.zeros`, but returns the tensor in
        compressed TTD format with all TT-ranks equal to 1.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new tensor, e.g. ``(2, 3)`` or ``2``.
        dtype : data-type, optional
            The desired data-type of the tensor, e.g. ``numpy.float64``.
            The default is ``numpy.float64``.

        Returns
        -------
        TTD
            Tensor of zeros with the given shape and dtype.

        See Also
        --------
        numpy.zeros : Equivalent function for dense arrays.
        TTD.ones : Return a new TTD of ones.
        TTD.full : Return a new TTD filled with a constant value.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.zeros((2, 2))
        >>> np.asarray(t)
        array([[0., 0.],
               [0., 0.]])

        """
        cores = [np.zeros((1, n, 1), dtype=dtype) for n in to_int_tuple(shape)]
        return TTD(cores)

    @staticmethod
    def full[DT: np.floating](
        shape: int | Sequence[int],
        fill_value: Scalar,
        *,
        dtype: np.dtype[DT] | None = None,
    ) -> TTD[DT]:
        """
        Return a new TTD representing a tensor filled with `fill_value`.

        Equivalent to :func:`numpy.full`, but returns the tensor in
        compressed TTD format with all TT-ranks equal to 1.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new tensor, e.g. ``(2, 3)`` or ``2``.
        fill_value : scalar
            Value used to fill the tensor.
        dtype : data-type, optional
            The desired data-type of the tensor, e.g. ``numpy.float64``.
            The default is ``numpy.float64``.

        Returns
        -------
        TTD
            Tensor filled with `fill_value`, with the given shape and dtype.

        See Also
        --------
        numpy.full : Equivalent function for dense arrays.
        TTD.ones : Return a new TTD of ones.
        TTD.zeros : Return a new TTD of zeros.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.full((2, 2), 10)
        >>> np.asarray(t)
        array([[10., 10.],
               [10., 10.]])

        """
        return TTD.ones(shape, dtype=dtype) * fill_value

    @override
    def __repr__(self) -> str:
        """Return a string representation of the TTD object."""
        return str(self)

    @override
    def __str__(self) -> str:
        """Return a string representation of the TTD object."""
        return f"TTD(shape={self.shape},\n{'\n\n'.join(map(str, self.data))})"

    @property
    @implements_function("shape")
    def shape(self) -> tuple[int, ...]:
        """
        Tuple of tensor dimensions.

        Equivalent to :attr:`numpy.ndarray.shape`. Unlike the NumPy
        attribute, it is read-only and cannot be assigned to reshape the
        tensor in place.

        Returns
        -------
        tuple of ints
            The shape of the uncompressed tensor, i.e. the size of each
            mode (physical dimension of each core).

        See Also
        --------
        TTD.ndim : Number of tensor dimensions.
        TTD.size : Number of elements in the uncompressed tensor.

        Examples
        --------
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 3, 4))
        >>> t.shape
        (2, 3, 4)

        """
        return tuple(core.shape[1] for core in self.data)

    @property
    @implements_function("ndim")
    def ndim(self) -> int:
        """
        Number of tensor dimensions.

        Equivalent to :attr:`numpy.ndarray.ndim`. Equal to the number of
        cores of the TTD.

        Returns
        -------
        int
            The number of dimensions of the uncompressed tensor.

        See Also
        --------
        TTD.shape : Tuple of tensor dimensions.
        TTD.size : Number of elements in the uncompressed tensor.

        Examples
        --------
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 3, 4))
        >>> t.ndim
        3

        """
        return len(self.data)

    @property
    @implements_function("size")
    def size(self) -> int:
        """
        Number of elements in the uncompressed tensor.

        Equivalent to :attr:`numpy.ndarray.size`. Equal to
        ``math.prod(self.shape)``, i.e. the product of the tensor's
        dimensions. This is the size of the dense tensor, not of the
        compressed representation; see :attr:`TTD.compressed_size` for
        the latter.

        Returns
        -------
        int
            The number of elements of the uncompressed tensor.

        See Also
        --------
        TTD.shape : Tuple of tensor dimensions.
        TTD.ndim : Number of tensor dimensions.
        TTD.compressed_size : Size of the compressed representation.

        Examples
        --------
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 3, 4))
        >>> t.size
        24

        """
        return math.prod(self.shape)

    @property
    def compressed_size(self) -> int:
        """Return the size of the compressed representation."""
        return sum(core.size for core in self.data)

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return the internal ranks of the TTD object."""
        return tuple(core.shape[2] for core in self.data[:-1])

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
            return np.asarray(reshaped, dtype=dtype, copy=copy)

        if copy is False:
            raise ValueError(
                "`copy=False` is supported only for TTD representing a "
                "single-dimensional array."
            )

        # Multiply all cores together. This is equivalent to a repeated
        # tensordot, but faster since einsum does compute order optimizations.
        # The generated expression looks like ABC,CDE,EFG,…, XYZ->ABCD…YZ, where
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

        # remove first and last singleton dimensions
        squeezed = result.squeeze((0, -1))

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

        # Suppose that 𝐀 is in the TTD format:
        # 𝐀(i₁, ..., i_d) = 𝐆₁(i₁) 𝐆₂(i₂) ... 𝐆_d(i_d)

        # (𝐆₁, ..., 𝐆_d)
        # Note: cores are 0 indexed here but 1 indexed in the paper, so 𝐆ₖ = G[k - 1]
        cores = self.data
        d = len(cores)

        # do a right-to-left qr-sweep
        orthogonalize_right(cores)

        delta = truncation_parameter(self, epsilon)

        for k in range(1, d):  # for k = 1 to d-1
            # G = 𝐆ₖ(αₖ₋₁; iₖβₖ)
            core = cores[k - 1]
            beta_k1, i_k, beta_k = core.shape
            # 𝐔, 𝚲, 𝐕ᵀ := SVDᵟ(𝐆ₖ(βₖ₋₁; iₖβₖ))
            u, s, v_t = delta_truncated_svd(core.reshape(beta_k1 * i_k, beta_k), delta)
            # 𝐆ₖ(βₖ₋₁; iₖγₖ) = 𝐔
            cores[k - 1] = u.reshape((beta_k1, i_k, -1))
            # 𝐆ₖ₊₁ := 𝐆ₖ₊₁ ×₁ (𝐕𝚲)ᵀ = 𝐕𝚲 ⋅ 𝐆ₖ₊₁
            cores[k] = dot_product(scale_matrix(s, v_t), cores[k])

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

        if not hasattr(ufunc, "__name__") or not isinstance(ufunc.__name__, str):  # pyright: ignore[reportUnnecessaryIsInstance]
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
        if not hasattr(func, "__name__") or not isinstance(func.__name__, str):  # pyright: ignore[reportUnnecessaryIsInstance]
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

    def copy(self) -> TTD[DType]:
        """
        Return a copy of the TTD.

        Equivalent to :meth:`numpy.ndarray.copy`, but copies the
        compressed representation: a new TTD with a copy of each core.
        Mutating the copy does not affect the original.

        Unlike :meth:`numpy.ndarray.copy`, there is no ``order`` parameter
        since the TTD format dictates the memory layout of the cores.

        Returns
        -------
        TTD
            A copy of the TTD.

        See Also
        --------
        numpy.copy : Similar function for dense arrays.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 2))
        >>> c = t.copy()
        >>> np.asarray(c)
        array([[1., 1.],
               [1., 1.]])

        """
        return self.__class__((a.copy() for a in self.data), dtype=self.dtype)

    def transpose(self, axes: tuple[int, ...] | None = None) -> TTD[DType]:
        """
        Permute the dimensions of the TTD.

        Equivalent to :meth:`numpy.ndarray.transpose` and
        :func:`numpy.transpose`. See :func:`wren_ttd.ops.transpose` for
        details about the implementation.

        Parameters
        ----------
        axes : tuple of ints or None, optional
            Permutation of the axes. If None, the axes are reversed.
            Otherwise, ``axes[i]`` is the axis of the input that becomes
            the ``i``-th axis of the result, so it must be a permutation
            of ``range(self.ndim)``.

        Returns
        -------
        TTD
            TTD with permuted axes.

        See Also
        --------
        numpy.transpose : Equivalent function for dense arrays.
        TTD.swapaxes : Swap two axes of the TTD.
        TTD.T : Reverse the order of the axes.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 3, 4))
        >>> t.transpose((2, 0, 1)).shape
        (4, 2, 3)
        >>> t.transpose().shape
        (4, 3, 2)

        """
        return ops.transpose(self, axes)

    def swapaxes(self, axis1: int, axis2: int) -> TTD[DType]:
        """
        Swap two axes of the TTD.

        Equivalent to :meth:`numpy.ndarray.swapaxes` and
        :func:`numpy.swapaxes`. If `axis1` and `axis2` are equal, the
        TTD is returned unchanged.

        Parameters
        ----------
        axis1 : int
            First axis to swap. May be negative, counting from the end.
        axis2 : int
            Second axis to swap. May be negative, counting from the end.

        Returns
        -------
        TTD
            TTD with `axis1` and `axis2` swapped.

        See Also
        --------
        numpy.swapaxes : Equivalent function for dense arrays.
        TTD.transpose : Permute the dimensions of the TTD.
        TTD.T : Reverse the order of the axes.

        Examples
        --------
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 3, 4))
        >>> t.swapaxes(0, 2).shape
        (4, 3, 2)

        """
        return ops.swapaxes(self, axis1, axis2)

    @property
    def T(self) -> TTD[DType]:  # noqa: N802
        """
        View of the TTD with reversed dimensions.

        Equivalent to :attr:`numpy.ndarray.T` and to calling
        ``self.transpose()`` with no arguments. Reversing the order of
        the cores is exact: no truncation is performed.

        Returns
        -------
        TTD
            TTD with reversed axes.

        See Also
        --------
        TTD.transpose : Permute the dimensions of the TTD.
        TTD.swapaxes : Swap two axes of the TTD.

        Examples
        --------
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 3, 4))
        >>> t.T.shape
        (4, 3, 2)

        """
        return TTD(reverse_cores(self.data), dtype=self.dtype)

    @overload
    def __getitem__(self, key: SupportsIndex) -> TTD[DType] | DType: ...

    @overload
    def __getitem__(self, key: slice[SupportsIndex | None]) -> TTD[DType]: ...

    @overload
    def __getitem__(self, key: EllipsisType) -> TTD[DType]: ...

    @overload
    def __getitem__(self, key: Sequence[Index1D]) -> TTD[DType] | DType: ...

    @override
    def __getitem__(
        self,
        key: Index1D | Sequence[Index1D] | EllipsisType,
    ) -> TTD[DType] | DType:
        """
        Return a subtensor of the TTD.

        Equivalent to indexing a :class:`numpy.ndarray` with basic
        indexing: each entry of `key` addresses the corresponding mode
        of the tensor. An integer index contracts away that mode, while
        a slice preserves it. Indexing all modes with integers returns
        a scalar; otherwise a TTD is returned. ``...`` (Ellipsis) alone
        returns a copy of the TTD.

        Only integer and slice indices are supported; boolean masks and
        advanced (fancy) indexing are not supported.

        Parameters
        ----------
        key : int, slice, Ellipsis, or sequence thereof
            Index for each addressed mode. A single ``int`` or ``slice``
            addresses the first mode; a sequence addresses one mode per
            entry. Negative indices count from the end of the mode.

        Returns
        -------
        TTD or scalar
            The indexed subtensor, or a scalar if every mode was indexed
            with an integer.

        See Also
        --------
        numpy.ndarray.__getitem__ : Equivalent indexing for dense arrays.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 3))
        >>> np.asarray(t[0])
        array([1., 1., 1.])
        >>> t[0, 1]
        np.float64(1.0)
        >>> t[...].shape
        (2, 3)

        """
        if key is Ellipsis:
            return self.copy()

        if isinstance(key, (SupportsIndex, slice)):
            return ops.get_item(self, (key,))

        if isinstance(key, Sequence):
            return ops.get_item(self, key)

        raise NotImplementedError

    @override
    def __len__(self) -> int:
        """
        Return the length of the first axis.

        Equivalent to ``len`` on a :class:`numpy.ndarray`, i.e. to
        ``self.shape[0]``: the size of the first mode of the tensor.

        Returns
        -------
        int
            The size of the first mode.

        See Also
        --------
        TTD.shape : Tuple of tensor dimensions.

        Examples
        --------
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((4, 2, 3))
        >>> len(t)
        4

        """
        return self.data[0].shape[1]

    @override
    def __add__(self, other: TTD[DType] | Scalar) -> TTD[DType]:
        """
        Add a TTD or scalar to this TTD, element-wise.

        Equivalent to :func:`numpy.add` and the ``+`` operator on
        :class:`numpy.ndarray`. If `other` is a scalar, it is broadcast
        to the shape of this TTD. The TT-ranks of the result are the
        sums of the operand ranks.

        Parameters
        ----------
        other : TTD or scalar
            The TTD or scalar to add. A TTD must have the same shape as
            this TTD.

        Returns
        -------
        TTD
            The element-wise sum.

        See Also
        --------
        numpy.add : Equivalent ufunc for dense arrays.
        TTD.__radd__ : Reflected addition.
        TTD.__iadd__ : In-place addition.
        TTD.__sub__ : Subtraction.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 2))
        >>> np.asarray(t + 2)
        array([[3., 3.],
               [3., 3.]])

        """
        return ops.add(self, other)

    @override
    def __iadd__(self, other: TTD[DType] | Scalar) -> TTD[DType]:
        """
        Add a TTD or scalar to this TTD in place, element-wise.

        Equivalent to the ``+=`` operator on :class:`numpy.ndarray` and
        to :func:`numpy.add` with ``out`` set to this TTD. See
        :meth:`TTD.__add__` for the semantics of the addition. The cores
        of this TTD are replaced with the cores of the sum.

        Parameters
        ----------
        other : TTD or scalar
            The TTD or scalar to add. A TTD must have the same shape as
            this TTD.

        Returns
        -------
        TTD
            This TTD, holding the element-wise sum.

        See Also
        --------
        numpy.add : Equivalent ufunc for dense arrays.
        TTD.__add__ : Addition.
        TTD.__radd__ : Reflected addition.

        """
        return ops.add(self, other, out=self)

    @override
    def __radd__(self, other: TTD[DType] | Scalar) -> TTD[DType]:
        """
        Add this TTD to a TTD or scalar, element-wise (reflected).

        Equivalent to :func:`numpy.add` with the operands swapped, i.e.
        it computes ``other + self``. Called when the left operand does
        not support the addition. Addition is commutative, so this is
        the same as :meth:`TTD.__add__`.

        Parameters
        ----------
        other : TTD or scalar
            The TTD or scalar to add to this TTD. A TTD must have the
            same shape as this TTD.

        Returns
        -------
        TTD
            The element-wise sum.

        See Also
        --------
        numpy.add : Equivalent ufunc for dense arrays.
        TTD.__add__ : Addition.
        TTD.__iadd__ : In-place addition.

        """
        return ops.add(other, self)

    @override
    def __sub__(self, other: TTD[DType] | Scalar) -> TTD[DType]:
        """
        Subtract a TTD or scalar from this TTD, element-wise.

        Equivalent to :func:`numpy.subtract` and the ``-`` operator on
        :class:`numpy.ndarray`. If `other` is a scalar, it is broadcast
        to the shape of this TTD. Computed as the addition of this TTD
        with the negated `other`.

        Parameters
        ----------
        other : TTD or scalar
            The TTD or scalar to subtract. A TTD must have the same
            shape as this TTD.

        Returns
        -------
        TTD
            The element-wise difference.

        See Also
        --------
        numpy.subtract : Equivalent ufunc for dense arrays.
        TTD.__rsub__ : Reflected subtraction.
        TTD.__isub__ : In-place subtraction.
        TTD.__add__ : Addition.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.full((2, 2), 3.0)
        >>> np.asarray(t - 1)
        array([[2., 2.],
               [2., 2.]])

        """
        return ops.add(self, -other)

    @override
    def __isub__(self, other: TTD[DType] | Scalar) -> TTD[DType]:
        """
        Subtract a TTD or scalar from this TTD in place, element-wise.

        Equivalent to the ``-=`` operator on :class:`numpy.ndarray` and
        to :func:`numpy.subtract` with ``out`` set to this TTD. See
        :meth:`TTD.__sub__` for the semantics of the subtraction. The
        cores of this TTD are replaced with the cores of the difference.

        Parameters
        ----------
        other : TTD or scalar
            The TTD or scalar to subtract. A TTD must have the same
            shape as this TTD.

        Returns
        -------
        TTD
            This TTD, holding the element-wise difference.

        See Also
        --------
        numpy.subtract : Equivalent ufunc for dense arrays.
        TTD.__sub__ : Subtraction.
        TTD.__rsub__ : Reflected subtraction.

        """
        return ops.add(self, -other, out=self)

    @override
    def __rsub__(self, other: TTD[DType] | Scalar) -> TTD[DType]:
        """
        Subtract this TTD from a TTD or scalar, element-wise (reflected).

        Equivalent to :func:`numpy.subtract` with the operands swapped,
        i.e. it computes ``other - self``. Called when the left operand
        does not support the subtraction.

        Parameters
        ----------
        other : TTD or scalar
            The TTD or scalar to subtract this TTD from. A TTD must have
            the same shape as this TTD.

        Returns
        -------
        TTD
            The element-wise difference ``other - self``.

        See Also
        --------
        numpy.subtract : Equivalent ufunc for dense arrays.
        TTD.__sub__ : Subtraction.
        TTD.__isub__ : In-place subtraction.

        """
        return ops.add(-self, other)

    @override
    def __mul__(self, other: TTD[DType] | Scalar) -> TTD[DType]:
        """
        Multiply this TTD by a TTD or scalar, element-wise.

        Equivalent to :func:`numpy.multiply` and the ``*`` operator on
        :class:`numpy.ndarray`. Multiplying by a scalar scales a single
        core. Multiplying by another TTD computes the Hadamard
        (element-wise) product, whose TT-ranks are the products of the
        operand ranks, and requires both operands to have the same
        shape.

        Parameters
        ----------
        other : TTD or scalar
            The TTD or scalar to multiply with. A TTD must have the same
            shape as this TTD.

        Returns
        -------
        TTD
            The element-wise product.

        See Also
        --------
        numpy.multiply : Equivalent ufunc for dense arrays.
        TTD.__rmul__ : Reflected multiplication.
        TTD.__imul__ : In-place multiplication.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.full((2, 2), 2.0)
        >>> np.asarray(t * 3)
        array([[6., 6.],
               [6., 6.]])

        """
        return ops.multiply(self, other)

    @override
    def __imul__(self, other: TTD[DType] | Scalar) -> TTD[DType]:
        """
        Multiply this TTD by a TTD or scalar in place, element-wise.

        Equivalent to the ``*=`` operator on :class:`numpy.ndarray` and
        to :func:`numpy.multiply` with ``out`` set to this TTD. See
        :meth:`TTD.__mul__` for the semantics of the multiplication. The
        cores of this TTD are replaced with the cores of the product.

        Parameters
        ----------
        other : TTD or scalar
            The TTD or scalar to multiply with. A TTD must have the same
            shape as this TTD.

        Returns
        -------
        TTD
            This TTD, holding the element-wise product.

        See Also
        --------
        numpy.multiply : Equivalent ufunc for dense arrays.
        TTD.__mul__ : Multiplication.
        TTD.__rmul__ : Reflected multiplication.

        """
        return ops.multiply(self, other, out=self)

    @override
    def __rmul__(self, other: TTD[DType] | Scalar) -> TTD[DType]:
        """
        Multiply a TTD or scalar with this TTD, element-wise (reflected).

        Equivalent to :func:`numpy.multiply` with the operands swapped,
        i.e. it computes ``other * self``. Called when the left operand
        does not support the multiplication. Multiplication is
        commutative, so this is the same as :meth:`TTD.__mul__`.

        Parameters
        ----------
        other : TTD or scalar
            The TTD or scalar to multiply with this TTD. A TTD must have
            the same shape as this TTD.

        Returns
        -------
        TTD
            The element-wise product.

        See Also
        --------
        numpy.multiply : Equivalent ufunc for dense arrays.
        TTD.__mul__ : Multiplication.
        TTD.__imul__ : In-place multiplication.

        """
        return ops.multiply(self, other)

    @override
    def __neg__(self) -> TTD[DType]:
        """
        Negate the TTD, element-wise.

        Equivalent to :func:`numpy.negative` and the unary ``-`` operator
        on :class:`numpy.ndarray`. Computed as multiplication by ``-1``.

        Returns
        -------
        TTD
            The negated TTD, i.e. ``-self`` element-wise.

        See Also
        --------
        numpy.negative : Equivalent ufunc for dense arrays.

        Examples
        --------
        >>> import numpy as np
        >>> from wren_ttd.core import TTD
        >>> t = TTD.ones((2, 2))
        >>> np.asarray(-t)
        array([[-1., -1.],
               [-1., -1.]])

        """
        return ops.neg(self)
