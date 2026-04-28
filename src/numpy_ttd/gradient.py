import numpy as np

from numpy_ttd.tt_matrix import TTMatrix
from numpy_ttd.types import Matrix, MatrixCore


def gradient_1d[T: np.floating](n: int, h: T | float, dtype: np.dtype[T]) -> Matrix[T]:
    """
    Build the 1D finite-difference matrix for the gradient.

    The resulting n × n matrix is in the format

        h * tridiagonal(-1, 1, 0).

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

    gradient = np.zeros((n, n), dtype=dtype)

    gradient += np.eye(n, k=-1)
    gradient -= np.eye(n)

    return np.divide(gradient, h)


def tt_gradient[DT: np.floating](
    shape: tuple[int, ...],
    h: float | DT | tuple[float | DT, ...],
    dtype: np.dtype[DT],
) -> TTMatrix[DT]:
    dimensions = len(shape)

    if isinstance(h, tuple):
        if len(h) != dimensions:
            raise ValueError("h must be a scalar or have the same length as shape")
        steps = [dtype.type(v) for v in h]
    else:
        steps = [dtype.type(h)] * dimensions

    cores: list[MatrixCore[DT]] = []

    def g(k: int) -> Matrix[DT]:
        return gradient_1d(shape[k], steps[k], dtype)

    def I(k: int) -> Matrix[DT]:
        return np.eye(shape[k], dtype=dtype)

    def J(k: int) -> Matrix[DT]:
        return np.eye(shape[k], k=1, dtype=dtype)

    def core(r_l: int, k: int, r_r: int) -> MatrixCore[DT]:
        n = shape[k]
        return np.zeros((r_l, n, n, r_r), dtype=dtype)

    if dimensions == 1:
        G: MatrixCore[DT] = core(1, 0, 1)
        G[0, :, :, 0] = g(0)
        return TTMatrix([G])

    first: MatrixCore[DT] = core(1, 0, 2)
    first[0, :, :, 0] = I(0)
    first[0, :, :, 1] = -J(0)

    cores.append(first)

    for i in range(1, dimensions - 1):
        middle: MatrixCore[DT] = core(2, i, 2)
        middle[0, :, :, 0] = I(i)
        middle[0, :, :, 1] = -J(i)
        middle[1, :, :, 1] = J(i).T

        cores.append(middle)

    last = core(2, -1, 1)
    last[0, :, :, 0] = J(-1) - I(k=-1)
    last[1, :, :, 0] = J(-1).T

    cores.append(last)

    print(*cores, sep="\n")

    return TTMatrix(cores)
