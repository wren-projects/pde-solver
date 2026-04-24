import numpy as np

from numpy_ttd.tt_matrix import TTMatrix
from numpy_ttd.types import Matrix, MatrixCore


def laplace_1d[T: np.floating](n: int, h: T | float, dtype: np.dtype[T]) -> Matrix[T]:
    """
    Build the 1D finite-difference matrix for the Laplacian.

    The resulting n × n matrix is in the format

        h⁻² * tridiagonal(-1, 2, -1).

    Parameters
    ----------
    n : int
        Number of grid points.
    h : float
        Spacing between grid points.
    dtype : np.dtype
        NumPy dtype, e.g. np.float64.

    Returns
    -------
    Matrix[T]
        The 1D finite-difference matrix.

    """
    if n <= 0:
        raise ValueError("n must be positive")

    if h <= 0:
        raise ValueError("h must be positive")

    laplacian = np.zeros((n, n), dtype=dtype)

    laplacian += np.eye(n, k=-1, dtype=dtype)
    laplacian += np.eye(n, k=1, dtype=dtype)
    laplacian -= 2 * np.eye(n, dtype=dtype)

    return np.divide(laplacian, h * h)


def tt_laplace[DT: np.floating](
    shape: tuple[int, ...],
    h: float | DT | tuple[float | DT, ...] = 1.0,
    *,
    dtype: np.dtype[DT],
) -> TTMatrix[DT]:
    """
    Build the full high-dimensional Laplacian in TT-matrix form.

    The resulting TT operator is in the format

        A = ∑ᵢᵏ I₁ ⊗ I₂ ⊗ … ⊗ Iᵢ₋₁ ⊗ Lᵢ ⊗ Iᵢ₊₁ ⊗ … ⊗ Iₖ,

    where Lᵢ is the 1D Laplacian on axis i.

    The shapes of the cores are
        first core:   1 × n₁ × n₁ × 2
        middle cores: 2 × nᵢ × nᵢ × 2
        last core:    2 × nₖ × nₖ × 1

    Parameters
    ----------
    shape:
        [n₁, n₂, ..., nₖ]
        Number of grid points in each physical dimension.

    h:
        Either one scalar spacing for all dimensions or a tuple with spacing for
        each dimension: (h₁, h₂, ..., hₖ)

    dtype:
        NumPy dtype, e.g. np.float64.

    Returns
    -------
    TTMatrix[T]
        The high-dimensional Laplacian in TT-matrix form.

    """
    if not shape:
        raise ValueError("shape must not be empty")

    dimensions = len(shape)  # number of dimensions

    if isinstance(h, tuple):
        if len(h) != dimensions:
            raise ValueError("h must be a scalar or have the same length as shape")
        steps = [dtype.type(v) for v in h]
    else:
        steps = [dtype.type(h)] * dimensions

    def L(k: int) -> Matrix[DT]:
        return laplace_1d(shape[k], steps[k], dtype)

    def I(k: int) -> Matrix[DT]:
        return np.eye(shape[k], dtype=dtype)

    def empty_core(r_l: int, k: int, r_r: int) -> MatrixCore[DT]:
        n = shape[k]
        return np.zeros((r_l, n, n, r_r), dtype=dtype)

    cores: list[MatrixCore[DT]] = []

    if dimensions == 1:
        G = empty_core(1, 0, 1)
        G[0, :, :, 0] = L(0)
        return TTMatrix([G])

    # First core: [L1, I1]
    G1 = empty_core(1, 0, 2)
    G1[0, :, :, 0] = L(0)
    G1[0, :, :, 1] = I(0)
    cores.append(G1)

    # Middle cores: [[Ik, 0], [Lk, Ik]]
    for k in range(1, dimensions - 1):
        Gk = empty_core(2, k, 2)

        Ik = I(k)
        Gk[0, :, :, 0] = Ik
        Gk[1, :, :, 0] = L(k)
        Gk[1, :, :, 1] = Ik

        cores.append(Gk)

    # Last core: [[ID], [LD]]
    GD = empty_core(2, -1, 1)
    GD[0, :, :, 0] = I(-1)
    GD[1, :, :, 0] = L(-1)
    cores.append(GD)

    return TTMatrix(cores)
