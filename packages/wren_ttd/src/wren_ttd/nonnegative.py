from __future__ import annotations
import time

import numpy as np
from wren_common.types import Matrix, NDArray, Scalar, Vector

from wren_ttd._helpers import expand_cores, ttsvd
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


def _right_grams[DT: np.floating](
    cores: list[Core[DT]], dtype: np.dtype[DT]
) -> list[Matrix[DT]]:
    """Compute the right Gram matrices Q^{>n} of a TT decomposition."""
    count = len(cores)
    grams: list[Matrix[DT]] = [np.ones((1, 1), dtype=dtype) for _ in range(count)]

    for n in range(count - 2, -1, -1):
        core = cores[n + 1]
        grams[n] = np.einsum("aic,cd,bid->ab", core, grams[n + 1], core)

    return grams


def _right_contractions[DT: np.floating](
    target_cores: list[Core[DT]], cores: list[Core[DT]]
) -> list[Matrix[DT]]:
    """
    Contract the right-hand parts of two TTs with each other.

    Entry ``[d, b]`` of the result for core n is the contraction of the target
    cores to the right of n (with left rank d) and the cores to the right of n
    (with left rank b).
    """
    count = len(cores)
    contractions: list[Matrix[DT]] = [
        np.ones((1, 1), dtype=cores[0].dtype) for _ in range(count)
    ]

    for n in range(count - 2, -1, -1):
        contractions[n] = np.einsum(
            "die,bif,ef->db",
            target_cores[n + 1],
            cores[n + 1],
            contractions[n + 1],
        )

    return contractions


def _left_gram[DT: np.floating](core: Core[DT], gram: Matrix[DT]) -> Matrix[DT]:
    """Extend a left Gram matrix by one core, computing Q^{<n+1} from Q^{<n}."""
    return np.einsum("cia,cd,dib->ab", core, gram, core)


def _left_contraction[DT: np.floating](
    target_core: Core[DT], core: Core[DT], contraction: Matrix[DT]
) -> Matrix[DT]:
    """Extend a left contraction by one pair of target and current cores."""
    return np.einsum("cid,aib,ca->db", target_core, core, contraction)


def _normalize_right_to_left[DT: np.floating](cores: list[Core[DT]]) -> None:
    """Normalize the cores from right to left (Algorithm 1, lines 3-6)."""
    for n in range(len(cores) - 1, 0, -1):
        core = cores[n]
        for r in range(core.shape[0]):
            norm = np.linalg.norm(core[r])
            if norm > 0:
                core[r] /= norm
                cores[n - 1][:, :, r] *= norm


def _hals_sweep[DT: np.floating](
    target_cores: list[Core[DT]], cores: list[Core[DT]], dtype: np.dtype[DT]
) -> None:
    """Run one complete right-to-left/left-to-right HALS sweep."""
    _normalize_right_to_left(cores)
    q_greater = _right_grams(cores, dtype)
    right = _right_contractions(target_cores, cores)

    q_less: Matrix[DT] = np.ones((1, 1), dtype=cores[0].dtype)
    left: Matrix[DT] = np.ones((1, 1), dtype=cores[0].dtype)

    for n, core in enumerate(cores):
        diagonals_less = np.diag(q_less)
        diagonals_greater = np.diag(q_greater[n])
        inverse_less = np.divide(
            1.0,
            diagonals_less,
            out=np.zeros_like(diagonals_less),
            where=diagonals_less > 0,
        )
        inverse_greater = np.divide(
            1.0,
            diagonals_greater,
            out=np.zeros_like(diagonals_greater),
            where=diagonals_greater > 0,
        )

        # W₁ = Y ×₁ G^{<n}ᵀ ×₃ G^{>n}ᵀ and W₂ = G^(n) ×₁ Q^{<n} ×₃ Q^{>n}.
        w1 = np.einsum("ca,cid,db->aib", left, target_cores[n], right[n])
        w2 = np.einsum("aA,AiB,bB->aib", q_less, core, q_greater[n])

        # Z^(n) = (W₁ - W₂) ×₁ D^{<n} ×₃ D^{>n}.
        z = (w1 - w2) * inverse_less[:, None, None] * inverse_greater[None, None, :]

        # Nonnegative HALS updates of the core fibres, updating Z after every
        # fibre with Equation (9).
        for a in range(core.shape[0]):
            for b in range(core.shape[2]):
                updated = np.maximum(0.0, core[a, :, b] + z[a, :, b])
                delta = updated - core[a, :, b]
                core[a, :, b] = updated

                z -= (
                    (q_less[:, a] * inverse_less)[:, None, None]
                    * delta[None, :, None]
                    * (q_greater[n][:, b] * inverse_greater)[None, None, :]
                )

        # Left-to-right normalization.
        if n < len(cores) - 1:
            for r in range(core.shape[2]):
                norm = np.linalg.norm(core[:, :, r])
                if norm > 0:
                    core[:, :, r] /= norm
                    cores[n + 1][r, :, :] *= norm

        left = _left_contraction(target_cores[n], core, left)
        q_less = _left_gram(core, q_less)


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

            array = expand_cores(cores)

        return TroppTTD(cores)

    def round(self, epsilon: np.floating | float = DEFAULT_EPSILON) -> None:
        """
        Round the TroppTTD object by decreasing ranks.

        Uses SVD for rank truncation and the NTT-HALS refinement to obtain
        nonnegative TT-cores while preserving the approximation accuracy.
        Ensures that the ranks of the rounded TTD 𝐀̃ are maximally reduced,
        while ensuring that the relative error is less than `epsilon`.

        Operates on the TTD object in-place. See also :func:`TTD.rounded` to get
        a rounded copy.

        Parameters
        ----------
        epsilon : float, optional
            The relative error tolerance for the compression. Uses system-wide default
            value if not provided.

        """
        # Y in Algorithm 1 is the tensor before the SVD rank truncation. Keep
        # its TT representation so the HALS sweeps stay in compressed form.
        target_cores = [core.copy() for core in self.data]

        super().round(epsilon)

        # Algorithm 1, line 1: initialize X by the (truncated) ALS solution and
        # take the absolute value of every core entry.
        svd_cores: list[Core[DType]] = [core.copy() for core in self.data]
        cores: list[Core[DType]] = [np.abs(core) for core in svd_cores]

        if len(cores) <= 1:
            self.data = cores
            return

        target_array = expand_cores(target_cores)

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
        max_iterations = 2000  # prevent infinite loops
        converged = False

        for _ in range(max_iterations):
            _hals_sweep(target_cores, cores, dtype=self.dtype)

            # stop once every entry satisfies the element wise tolerance promised
            # by rounding. Keep a margin for the error already present in the
            # target approximation.
            difference = expand_cores(cores) - target_array
            converged = np.all(
                np.abs(difference) <= 0.1 * epsilon * (1.0 + np.abs(target_array))
            )

            if converged:
                break

        # HALS cannot always represent the target with nonnegative cores at the
        # truncated ranks. In that case keep the more accurate SVD solution.
        self.data = cores if converged else svd_cores
