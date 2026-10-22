from typing import Protocol, runtime_checkable

import numpy as np
from wren_common.types import Matrix, Scalar, Vector

from wren_ttd.math import DEFAULT_EPSILON

type SVDResult[DT: np.floating] = tuple[Matrix[DT], Vector[DT], Matrix[DT]]


@runtime_checkable
class SVD[DT: np.floating, **Params](Protocol):
    def __call__(
        self,
        matrix: Matrix[DT],
        delta: Scalar = DEFAULT_EPSILON,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> SVDResult[DT]: ...


def delta_truncated_svd[DT: np.floating](
    matrix: Matrix[DT], delta: Scalar = DEFAULT_EPSILON
) -> tuple[Matrix[DT], Vector[DT], Matrix[DT]]:
    """
    Compute the SVD of a matrix, dropping singular values below `delta`.

    At least one singular value is always kept.

    Parameters
    ----------
    matrix : Matrix[DT]
        The matrix to decompose.
    delta : np.floating | float, optional
        The cutoff below which singular values are dropped, by default
        DEFAULT_EPSILON.

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

    # Always keep at least one singular value
    if not np.any(mask):
        return u[:, :1], s[:1], v_t[:1, :]

    return u[:, mask], s[mask], v_t[mask, :]


assert isinstance(delta_truncated_svd, SVD)
