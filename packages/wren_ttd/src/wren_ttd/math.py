from __future__ import annotations

import numpy as np
from wren_common.types import Matrix

DEFAULT_EPSILON = np.float64(1e-10)


def qr_columns[DT: np.floating](matrix: Matrix[DT]) -> tuple[Matrix[DT], Matrix[DT]]:
    """
    Compute the QR decomposition of a matrix with orthogonal columns.

    Parameters
    ----------
    matrix : Matrix[DT]
        The matrix to decompose.

    Returns
    -------
    tuple[Matrix[DT], Matrix[DT]]
        The QR decomposition in the form of matrix Q and matrix R.

    """
    return np.linalg.qr(matrix)


def qr_rows[DT: np.floating](matrix: Matrix[DT]) -> tuple[Matrix[DT], Matrix[DT]]:
    """
    Compute the QR decomposition of a matrix with orthogonal rows.

    Parameters
    ----------
    matrix : Matrix[DT]
        The matrix to decompose.

    Returns
    -------
    tuple[Matrix[DT], Matrix[DT]]
        The QR decomposition in the form of matrix Q and matrix R.

    """
    q, r = np.linalg.qr(matrix.T)

    return q.T, r.T
