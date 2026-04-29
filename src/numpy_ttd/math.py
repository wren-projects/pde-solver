from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from numpy_ttd.types import Matrix, Vector

DEFAULT_EPSILON = np.float64(1e-9)


def truncation_parameter[DT: np.floating](
    tensor: NDArray[DT], epsilon: np.floating | float = DEFAULT_EPSILON
) -> DT:
    """
    Compute the truncation parameter of a tensor.

    It uses the formula δ = (ε / √(d - 1)) ⋅ ‖A‖ᶠ.

    Parameters
    ----------
    tensor : NDArray[DT]
        The tensor to compute the truncation parameter of.
    epsilon : np.floating | float, optional
        The error tolerance for the compression, by default DEFAULT_EPSILON.

    Returns
    -------
    DT
        The truncation parameter.

    Raises
    ------
    ValueError
        If the tensor has less than 2 dimensions.

    """
    d = tensor.ndim
    if d <= 1:
        raise ValueError("Tensor must be at least 2D")

    return tensor.dtype.type(epsilon / math.sqrt(d - 1)) * np.linalg.norm(tensor)


def delta_truncated_svd[DT: np.floating](
    matrix: Matrix[DT], delta: np.floating | float = DEFAULT_EPSILON
) -> tuple[Matrix[DT], Vector[DT], Matrix[DT]]:
    """
    Compute δ-truncated Singular Value Decomposition of a matrix.

    Regular SVD decomposes a matrix into the matrix product 𝐔Σ𝐕ᵀ, where Σ is a
    diagonal matrix with singular values on its main diagonal and 𝐔 and 𝐕ᵀ are
    unitary matrices. δ-truncation additionally cuts off singular values smaller
    or equal to the parameter δ.

    Parameters
    ----------
    matrix : Matrix[DT]
        The matrix to compute the SVD of.
    delta : np.floating | float, optional
        The truncation parameter, by default DEFAULT_EPSILON.

    Returns
    -------
    tuple[Matrix[DT], Vector[DT], Matrix[DT]]
        The SVD in the form of matrix 𝐔, vector 𝐒 and matrix 𝐕ᵀ.

        Vector 𝐒 is the main diagonal of the δ-truncated matrix Σ. Matrices 𝐔
        and 𝐕ᵀ are truncated such as to match the shape of Σ.

    """
    u, s, v_t = np.linalg.svd(matrix, full_matrices=False)

    # Keep only singular values >= delta
    mask = s >= delta
    return u[:, mask], s[mask], v_t[mask, :]


def qr_rows[DT: np.floating](matrix: Matrix[DT]) -> tuple[Matrix[DT], Matrix[DT]]:
    """
    Compute the QR decomposition, where Q has orthogonal rows.

    Parameters
    ----------
    matrix : Matrix[DT]
        The matrix to compute the QR decomposition of.

    Returns
    -------
    tuple[Matrix[DT], Matrix[DT]]
        The QR decomposition in the form of matrix Q and matrix R.

    """
    q, r = np.linalg.qr(matrix.T)

    return q.T, r.T
