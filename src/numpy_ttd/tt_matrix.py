from collections.abc import Iterable
from typing import Any, final, overload

import numpy as np
import numpy.typing as npt

from numpy_ttd import TTD
from numpy_ttd.types import Core, MatrixCore, NDArray


def _to_3d_core[DType: np.floating](core: MatrixCore[DType]) -> Core[DType]:
    return core.reshape(core.shape[0], -1, core.shape[3])


@final
class TTMatrix[DType: np.floating]:
    """Class for storing matrices in TT format."""

    __slots__ = ("columns", "rows", "ttd")

    ttd: TTD[DType]
    rows: tuple[int, ...]
    columns: tuple[int, ...]

    def __init__(self, data: Iterable[MatrixCore[DType]]):
        """
        Create a new TT-matrix.

        Parameters
        ----------
        data : Iterable[MatrixCore[DType]]
            The cores of the TT-matrix.
        shape : tuple[int, ...]
            The shape of the TT-matrix.

        """
        cores = list(data)
        self.rows = tuple(core.shape[1] for core in cores)
        self.columns = tuple(core.shape[2] for core in cores)
        self.ttd = TTD(map(_to_3d_core, cores))

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
        # Inflate the inner TTD and reshape it back into the original matrix
        return np.asarray(self.ttd, dtype=dtype, copy=copy).reshape(
            # for rows n, m and columns i, j produces (n, i, m, j)
            sum(zip(self.rows, self.columns, strict=True), start=())
        )
