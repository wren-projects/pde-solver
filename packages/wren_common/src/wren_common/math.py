from __future__ import annotations

from typing import cast, overload

import numpy as np

from wren_common.types import Matrix, Vector


@overload
def dot_product[DT: np.dtype, A: int, B: int, C: int](
    a: np.ndarray[tuple[A, B], DT],
    b: np.ndarray[tuple[B, C], DT],
) -> np.ndarray[tuple[A, C], DT]: ...


@overload
def dot_product[DT: np.dtype, A: int, B: int, C: int, D: int](
    a: np.ndarray[tuple[A, B, C], DT],
    b: np.ndarray[tuple[C, D], DT],
) -> np.ndarray[tuple[A, B, D], DT]: ...


@overload
def dot_product[DT: np.dtype, A: int, B: int, C: int, D: int](
    a: np.ndarray[tuple[A, B], DT],
    b: np.ndarray[tuple[B, C, D], DT],
) -> np.ndarray[tuple[A, C, D], DT]: ...


@overload
def dot_product[DT: np.dtype, A: int, B: int, C: int, D: int, E: int](
    a: np.ndarray[tuple[A, B, C], DT],
    b: np.ndarray[tuple[C, D, E], DT],
) -> np.ndarray[tuple[A, B, D, E], DT]: ...


def dot_product[DT: np.dtype](
    a: np.ndarray[tuple[int, ...], DT], b: np.ndarray[tuple[int, ...], DT]
) -> np.ndarray[tuple[int, ...], DT]:
    """Compute the tensor dot product."""
    return cast(np.ndarray[tuple[int, ...], DT], np.tensordot(a, b, axes=1))


def scale_matrix[DT: np.floating](a: Vector[DT], b: Matrix[DT]) -> Matrix[DT]:
    """
    Scale a matrix by a vector.

    Equivalent to `np.diag(a) @ b`.

    Parameters
    ----------
    a : Vector[DT]
        The vector to scale the matrix by.
    b : Matrix[DT]
        The matrix to scale.

    Returns
    -------
    Matrix[DT]
        The scaled matrix.

    """
    return np.multiply(a[:, None], b)
