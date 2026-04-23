from collections.abc import Iterable
from typing import Any, final, overload, override

import numpy as np
import numpy.typing as npt

from numpy_ttd import TTD
from numpy_ttd.types import Core, Matrix, MatrixCore, NDArray


@final
class TTMatrix[DType: np.floating]:
    """Class for storing operators in TT format."""

    __slots__ = ("data", "dtype", "shape")

    data: list[MatrixCore[DType]]
    dtype: np.dtype[DType]
    shape: tuple[int, ...]

    def __init__(self, data: Iterable[MatrixCore[DType]], shape: tuple[int, ...]):
        self.data = list(data)
        self.dtype = self.data[0].dtype
        self.shape = shape

    @property
    def row_shape(self) -> tuple[int, ...]:
        return tuple(core.shape[1] for core in self.data)

    @property
    def col_shape(self) -> tuple[int, ...]:
        return tuple(core.shape[2] for core in self.data)

    @overload
    def __array__(
        self, dtype: None = None, *, copy: bool | None = None
    ) -> NDArray[DType]: ...

    @overload
    def __array__[DT: np.floating](
        self, dtype: np.dtype[DT], *, copy: bool | None = None
    ) -> NDArray[DT]: ...

    def __array__(
        self, dtype: npt.DTypeLike | None = None, *, copy: bool | None = None
    ) -> NDArray[Any]:
        """
        Expand a TT-matrix into a full NDArray.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        copy : bool, optional
            For compatibility with NumPy. Supports only `None` or `True`.

        Returns
        -------
        NDArray[DType]
            The matrix as a 2D :class:`numpy.ndarray` with the specified `dtype`.

        """
        if dtype is None:
            dtype = self.dtype

        # Empty TTD
        if not self.data:
            return np.empty((0,), dtype=dtype)

        if copy is False:
            raise ValueError("`copy=False` is not supported for TT-matrices.")

        # Multiply all cores together. This is equivalent to reduce(tensordot,
        # self.data), but faster since einsum does compute order optimizations
        # (at the cost of slightly uglier code).

        summation_indices: list[Any] = [
            item
            for i, core in enumerate(self.data)
            for item in (core, (3 * i, 3 * i + 1, 3 * i + 2, 3 * i + 3))
        ]

        result = cast(
            NDArray[DType],
            np.einsum(*summation_indices, optimize=True),  # pyright: ignore[reportAny]
        )

        # remove singleton dimensions
        squeezed = result.squeeze()

        return squeezed.astype(dtype)

    def apply(self, vector: TTD[DType]) -> TTD[DType]:
        """
        Multiply a TT-matrix with a TT-tensor.

        Parameters
        ----------
        vector : NDArray[DType]
            The vector to multiply with the TT-matrix.

        Returns
        -------
        NDArray[DType]
            The result of the multiplication -- a TT-tensor with the same shape
            as the input.

        """
        if len(self.data) != len(vector.data):
            raise ValueError("Number of cores must match.")

        def multiply_core(a: MatrixCore[DType], b: Core[DType]) -> Core[DType]:
            r_prev, i, j, r_next = a.shape
            s_prev, j, s_next = b.shape

            return np.einsum("rijR,sjS->rsiRS", a, b, optimize=True).reshape(
                r_prev * s_prev, i, r_next * s_next
            )

        return TTD(map(multiply_core, self.data, vector.data))

    def __matmul__(self, other: TTD[DType]) -> TTD[DType]:
        return self.apply(other)

    def __rmatmul__(self, other: TTD[DType]) -> TTD[DType]:
        return other.apply(self)
