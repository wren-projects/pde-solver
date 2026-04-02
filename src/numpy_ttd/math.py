from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from numpy_ttd.types import Core, Matrix, Vector

DEFAULT_EPSILON = np.float64(1e-9)


def frobenius_norm[DT: np.floating](tensor: NDArray[DT]) -> DT:
    """Compute the frobenius norm of an arbitrary tensor."""
    return np.sqrt(np.sum(np.square(tensor)))  # pyright: ignore[reportAny]


def truncation_parameter[DT: np.floating](
    tensor: NDArray[DT], epsilon: np.floating | float
) -> DT:
    """Compute the truncation parameter of a given tensor."""
    assert tensor.ndim > 1, "Tensor must be at least 2D"

    # δ = (ε / √(d - 1)) ⋅ ‖A‖ᶠ = ‖A‖ᶠ ⋅ (ε / √(d - 1))
    return frobenius_norm(tensor) * (epsilon / math.sqrt(tensor.ndim - 1))


def delta_truncated_svd[DT: np.floating](
    matrix: Matrix[DT], delta: np.floating | float = DEFAULT_EPSILON
) -> tuple[Matrix[DT], Vector[DT], Matrix[DT]]:
    """
    Compute SVD of a given matrix.

    Parameters
    ----------
    matrix : Matrix[DT]
        The matrix to compute the SVD of.
    delta : np.floating | float, optional
        The truncation parameter, by default DEFAULT_EPSILON.

    Returns
    -------
    tuple[Matrix[DT], Vector[DT], Matrix[DT]]
        The δ-truncated SVD of the matrix 𝐔, 𝐒 and 𝐕ᵀ.

        𝐒 is just the diagonal, with singular values >= delta. 𝐔 and 𝐕ᵀ
        are truncated accordingly.

    """
    u, s, v_t = np.linalg.svd(matrix, full_matrices=False)

    # Keep only singular values >= delta
    mask = s >= delta
    return u[:, mask], s[mask], v_t[mask, :]
