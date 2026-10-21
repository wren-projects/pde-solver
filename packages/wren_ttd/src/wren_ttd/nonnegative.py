from __future__ import annotations

import numpy as np
from numpy.lib.array_utils import normalize_axis_index
from wren_common.types import Matrix, NDArray, Scalar, Vector

from wren_ttd._helpers import ttsvd, unfold
from wren_ttd._svd import delta_truncated_svd
from wren_ttd.core import TTD
from wren_ttd.math import DEFAULT_EPSILON, qr_columns
from wren_ttd.types import Core

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
        iterations: int = 100
        column_samples: int = 38
        row_samples: int = 100

        cores: list[Core[DT]] = []

        # don't mutate the caller's array with the non-negativity projection
        array = array.copy()

        # repeat for specified number of iterations
        for _ in range(iterations):
            # X = max(A, 0)
            array[array < 0] = 0

            # G₁, G₂, …, Gₙ = TTSVD(A, Tropp, δ, k, l)
            cores = ttsvd(
                array,
                tropp_truncated_svd,
                epsilon,
                column_samples,
                row_samples,
            )

            array = TTD(cores).as_array()

        return TroppTTD(cores)

    def round(self, epsilon: np.floating | float = DEFAULT_EPSILON) -> None:
        """
        Round the TroppTTD object by decreasing ranks.

        Uses SVD for compression. Ensures that the ranks of the rounded
        TTD 𝐀̃ are maximally reduced, while ensuring that the relative error is
        less than `epsilon`.

        Operates on the TTD object in-place. See also :func:`TTD.rounded` to get
        a rounded copy.

        Parameters
        ----------
        epsilon : float, optional
            The relative error tolerance for the compression. Uses system-wide default
            value if not provided.

        """
        super().round(epsilon)

        """
Algorithm 1. NTT-HALS for the NTT Decomposition
Input: Y ∈ RI1×···×IN
+ , TT-ranks Rn, n = 1, . . . , N − 1.
Output: NTT tensor X in (2) with nonnegative TT-cores G(n).
1 Initialize X by ALS [6]. Set g(n)
rn−1,in,rn ← |g(n)
rn−1,in,rn | and G<1 = G>N = 1.
2 repeat
3 for n = N, N − 1, . . . , 2 do
4 Normalization: for each rn−1 = 1, . . . , Rn−1, update
G(n)
rn−1,:,: ← G(n)
rn−1,:,:/‖G(n)
rn−1,:,:‖F, G(n−1)
:,:,rn−1 ← G(n−1)
:,:,rn−1 · ‖G(n)
rn−1,:,:‖F.
5 Compute Q>n−1 ≡ G>n−1
(1) G>n−1
(1) recursively.
6 end
7 for n = 1, 2, . . . , N do
8 Compute Q<n ≡ G<n
(n)G<n
(n) recursively.
9 Let D<n = diag(diag(Q<n))−1, D>n = diag(diag(Q>n))−1.
10 Compute W(n)
1 , W(n)
2 ∈ RRn−1×In×Rn
+ by
W(n)
1,(2) = Y(n)
(
G>n
(1) ⊗ G<n
(n)
)
and W(n)
2 = G(n) ×1 Q<n ×3 Q>n.
11 Calculate Z(n) =
(
W(n)
1 − W(n)
2
)
×1 D<n ×3 D>n as in (7).
12 for rn−1 = 1, . . . , Rn−1, rn = 1, . . . , Rn do
13 g(n)
rn−1,:,rn ← [g(n)
rn−1,:,rn + z(n)
rn−1,:,rn ]+
14 Update Z(n) by (9).
15 end
16 Normalization: If n < N , for each rn = 1, . . . , Rn, update
G(n)
:,:,rn ← G(n)
:,:,rn /‖G(n)
:,:,rn ‖F, G(n+1)
rn,:,: ← G(n+1)
rn,:,: · ‖G(n)
:,:,rn ‖F.
17 end
18 until a stopping criterion is met;
        """
        # TODO: implement the NTT-HALS refinement described above. Until then,
        # the SVD-based rounding from `TTD.round` is the whole implementation.
