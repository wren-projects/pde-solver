from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, cast

import numpy as np

from numpy_ttd.types import Core, MatrixCore, NDArray

if TYPE_CHECKING:
    from numpy_ttd.tt_matrix import TTMatrix
    from numpy_ttd.ttd import TTD


def add[DType: np.floating](
    a: TTD[DType],
    b: TTD[DType],
    *,
    out: TTD[DType] | None = None,
) -> TTD[DType]:
    """Add two TTD objects."""
    # the import has to be here to avoid circular imports
    from numpy_ttd.ttd import TTD  # noqa: PLC0415

    if a.shape != b.shape:
        raise ValueError("Tensors with different shapes cannot be added.")

    if out is not None and out.shape != a.shape:
        raise ValueError("Output tensor has an incorrect shape.")

    cores = _add_cores(a.data, b.data, a.dtype)

    if out is None:
        return TTD(cores, dtype=a.dtype)

    out.data = list(cores)
    return out


def _add_cores[DType: np.floating](
    a: list[Core[DType]],
    b: list[Core[DType]],
    dtype: np.dtype,
) -> list[Core[DType]]:
    # Add vectors
    if len(a) == len(b) == 1:
        return [np.add(a[0], b[0])]

    def merge_cores(core_a: Core[DType], core_b: Core[DType]) -> Core[DType]:
        am, rank, an = core_a.shape
        bm, _, bn = core_b.shape

        result = np.zeros((am + bm, rank, an + bn), dtype=dtype)
        # upper left block
        result[:am, :, :an] = core_a
        # lower right block
        result[am:, :, an:] = core_b
        return result

    return [
        # stack first cores horizontally
        np.concatenate((a[0], b[0]), axis=2),
        # merge middle cores into blocks
        *map(merge_cores, a[1:-1], b[1:-1]),
        # stack last cores vertically
        np.concatenate((a[-1], b[-1]), axis=0),
    ]


def scalar_mul[DType: np.floating](
    a: TTD[DType],
    b: np.floating | float,
    out: TTD[DType] | None = None,
) -> TTD[DType]:
    """Multiply a TTD object by a scalar."""
    cores = a.data.copy()

    # find smallest core
    _, index = min((prod(core.shape), index) for index, core in enumerate(cores))

    cores[index] = np.multiply(cores[index], b)

    if out is not None:
        out.data = cores
        return out

    return a.__class__(cores, dtype=a.dtype)


def neg[DType: np.floating](
    a: TTD[DType],
) -> TTD[DType]:
    """Negate a TTD object."""
    return scalar_mul(a, -1.0)


def matvec[DType: np.floating](
    matrix: TTMatrix[DType], vector: TTD[DType]
) -> TTD[DType]:
    """
    Multiply a TT-matrix with a TT-tensor.

    Parameters
    ----------
    matrix : TTMatrix[DType]
        The TT-matrix to multiply.
    vector : NDArray[DType]
        The vector to multiply by the TT-matrix.

    Returns
    -------
    NDArray[DType]
        The result of the multiplication -- a TT-tensor with the same shape
        as the input vector.

    """
    from numpy_ttd.ttd import TTD  # noqa: PLC0415

    if len(matrix.ttd.data) != len(vector.data):
        raise ValueError(
            f"Missmatched number of cores: {len(matrix.ttd.data)} != {len(vector.data)}."
        )

    def multiply_cores(a: Core[DType], i: int, j: int, b: Core[DType]) -> Core[DType]:
        r_prev, _, r_next = a.shape
        s_prev, _, s_next = b.shape

        print(a.shape, b.shape, i, j)

        return cast(
            NDArray[DType],
            np.einsum(
                "rijR,sjS->rsiRS", a.reshape(r_prev, i, j, r_next), b, optimize=True
            ),
        ).reshape(r_prev * s_prev, i, r_next * s_next)

    return TTD(
        map(multiply_cores, matrix.ttd.data, matrix.rows, matrix.columns, vector.data)
    )
