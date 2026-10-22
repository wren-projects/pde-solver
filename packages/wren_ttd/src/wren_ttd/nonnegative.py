from __future__ import annotations

from typing import cast

import numpy as np
from wren_common.types import Matrix, NDArray, Scalar, Vector

from wren_ttd._helpers import ttsvd, truncation_parameter
from wren_ttd._svd import delta_truncated_svd
from wren_ttd.core import TTD
from wren_ttd.math import DEFAULT_EPSILON, qr_columns

__all__ = ["TroppTTD"]


def rademacher_matrix[DT: np.floating](
    rng: np.random.Generator, shape: tuple[int, int], dtype: np.dtype[DT]
) -> Matrix[DT]:
    sign_values: tuple[DT, DT] = tuple(map(dtype.type, (-1, 1)))

    return rng.choice(sign_values, size=shape)


def tropp_truncated_svd[DT: np.floating](
    matrix: Matrix[DT],
    delta: np.floating | float,
    column_samples: int,
    row_samples: int,
) -> tuple[Matrix[DT], Vector[DT], Matrix[DT]]:
    """
    Compute the SVD of a matrix, using the Tropp truncation algorithm.

    At least one singular value is always kept.

    Parameters
    ----------
    matrix : Matrix[DT]
        The matrix to decompose.
    delta : np.floating | float, optional
        The cutoff below which singular values are dropped, by default
        DEFAULT_EPSILON.
    column_samples : int
        The number of columns of the test matrix. Called `k` in the paper.
    row_samples : int
        The number of rows of the test matrix. Called `l` in the paper.

    Returns
    -------
    tuple[Matrix[DT], Vector[DT], Matrix[DT]]
        The SVD in the form of matrix 𝐔, vector 𝐒 and matrix 𝐕ᵀ.

        Vector 𝐒 is the main diagonal of the truncated matrix Σ. Matrices 𝐔
        and 𝐕ᵀ are truncated such as to match the shape of Σ.

    """
    rows, columns = matrix.shape

    rng = np.random.default_rng()

    # Ψ = TestMatrix(n, k)
    # Φ = TestMatrix(l, m)
    right_test_matrix = rademacher_matrix(rng, (columns, column_samples), matrix.dtype)
    left_test_matrix = rademacher_matrix(rng, (row_samples, rows), matrix.dtype)

    # Z = XΨ
    column_sketch: Matrix[DT] = matrix @ right_test_matrix
    # Q, R = QR(Z)
    column_basis, _ = qr_columns(column_sketch)
    # W = ΦQ
    projected_sketch: Matrix[DT] = left_test_matrix @ column_basis
    # P, T = QR(W)
    projected_basis, triangular_factor = qr_columns(projected_sketch)
    # G = T⁻¹ Pᵀ ΦX
    compressed_matrix: Matrix[DT] = (
        np.linalg.inv(triangular_factor) @ projected_basis.T @ left_test_matrix @ matrix
    )
    # Uᵣ, Σᵣ, Vᵣᵀ = SVD(G)
    left_singular_vectors, singular_values, right_singular_vectors_t = (
        delta_truncated_svd(compressed_matrix, delta)
    )
    # Uᵣ = Qᵣ Uᵣ
    left_singular_vectors: Matrix[DT] = column_basis @ left_singular_vectors

    return left_singular_vectors, singular_values, right_singular_vectors_t


class TroppTTD[DType: np.floating](TTD):
    """Nonnegative TTD using the Tropp algorithm."""

    def __init__(self, data, dtype=None):
        super().__init__(data, dtype=dtype)

    @staticmethod
    def from_ndarray[DT: np.floating](
        array: NDArray[DT], epsilon: Scalar = DEFAULT_EPSILON
    ) -> TroppTTD[DT]:
        """
        Compress an NDArray into a TTD object using the Tropp algorithm.

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
        column_samples: int = 38
        row_samples: int = 100

        delta = truncation_parameter(array, epsilon)

        return TroppTTD(
            ttsvd(
                array,
                tropp_truncated_svd,
                delta,
                column_samples,
                row_samples,
            ),
        )
