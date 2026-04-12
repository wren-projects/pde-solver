from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from numpy_ttd.types import Matrix, Vector

DEFAULT_EPSILON = np.float64(1e-9)


def truncation_parameter[DT: np.floating](
    tensor: NDArray[DT], epsilon: np.floating | float
) -> DT:
    """Compute the truncation parameter of a tensor."""
    assert tensor.ndim > 1, "Tensor must be at least 2D"

    # δ = (ε / √(d - 1)) ⋅ ‖A‖ᶠ = ‖A‖ᶠ ⋅ (ε / √(d - 1))
    return np.linalg.norm(tensor) * (epsilon / math.sqrt(tensor.ndim - 1))


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
