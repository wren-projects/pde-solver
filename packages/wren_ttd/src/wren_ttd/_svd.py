from typing import Protocol, runtime_checkable

import numpy as np
from wren_common.types import Matrix, Scalar, Vector

from wren_ttd.math import DEFAULT_EPSILON

type SVDResult[DT: np.floating] = tuple[Matrix[DT], Vector[DT], Matrix[DT]]


@runtime_checkable
class TruncatedSVD[DT: np.floating, **Params](Protocol):
    """
    Protocol for truncated SVD implementation.

    Parameters
    ----------
    matrix : Matrix[DT]
        The matrix to decompose.
    delta : Scalar, optional
        The cutoff below which singular values are dropped, by default
        DEFAULT_EPSILON.
    *args : Params.args
        Additional positional arguments to pass to the SVD implementation.
    **kwargs : Params.kwargs
        Additional keyword arguments to pass to the SVD implementation.

    Returns
    -------
    tuple[Matrix[DT], Vector[DT], Matrix[DT]]
        The SVD in the form of matrix 𝐔, vector 𝐒 and matrix 𝐕ᵀ.

        Vector 𝐒 is the main diagonal of the truncated matrix Σ. Matrices 𝐔
        and 𝐕ᵀ are truncated such as to match the shape of Σ.

    """

    def __call__(
        self,
        matrix: Matrix[DT],
        delta: Scalar,
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
    delta : Scalar, optional
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

    # Keep only singular values >= delta. SVD orders them in descending order,
    # so the kept ones are a prefix and can be simply sliced off.
    rank = max(int(np.count_nonzero(s >= delta)), 1)  # always keep at least one

    return u[:, :rank], s[:rank], v_t[:rank, :]


assert isinstance(delta_truncated_svd, TruncatedSVD)
