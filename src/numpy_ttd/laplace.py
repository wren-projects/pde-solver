from dataclasses import dataclass
import numpy as np
from numpy_ttd.types import Matrix, Core, MatrixCore


@dataclass
class TTMatrix[T: np.floating]:
    """Class for storing operators in TT format."""

    cores: list[MatrixCore[T]]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the TTMatrix object."""
        return tuple(core.shape[1] for core in self.data)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the TTMatrix object."""
        return len(self.cores)

    @property
    def col_shape(self) -> tuple[int, ...]:
        return tuple(core.shape[2] for core in self.cores)


def laplace_1d[T: np.floating](n: int, h: float, dtype: np.dtype[T]) -> Matrix[T]:
    """
    Build the 1D finite-difference matrix for -d^2/dx^2
    with the stencil (1/h^2) * tridiag(-1, 2, -1).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if h <= 0:
        raise ValueError("h must be positive")

    L = np.zeros((n, n), dtype=dtype)

    diag = 2.0 / (h * h)
    offdiag = -1.0 / (h * h)

    for i in range(n):
        L[i, i] = diag
        if i > 0:
            L[i, i - 1] = offdiag
        if i < n - 1:
            L[i, i + 1] = offdiag

    return L


def tt_laplacian[T: np.floating](
    shape: list[int],
    h: float | list[float],
    dtype: np.dtype[T] = np.float64,
) -> TTMatrix[T]:
    """
    Build the full high-dimensional Laplacian in TT-matrix form.

    Input
    -----
    shape:
        [n1, n2, ..., nD]
        Number of grid points in each physical dimension.

    h:
        Either one scalar spacing for all dimensions,
        or a list [h1, ..., hD].

    dtype:
        NumPy dtype, e.g. np.float64.

    Output
    ------
    TTMatrix[T]
        TT operator representing

            A = sum_k I ⊗ ... ⊗ L_k ⊗ ... ⊗ I

        where L_k is the 1D Laplacian on axis k.

    Core shapes
    -----------
    first core:  (1, n1, n1, 2)
    middle core: (2, nk, nk, 2)
    last core:   (2, nD, nD, 1)
    """
    if len(shape) == 0:
        raise ValueError("shape must not be empty")

    D = len(shape)  # number of dimensions

    if isinstance(h, (int, float)):
        steps = [float(h)] * D
    else:
        steps = [float(v) for v in h]
        if len(steps) != D:
            raise ValueError("h must be a scalar or have the same length as shape")

    Ls: list[Matrix[T]] = []
    Is: list[Matrix[T]] = []

    for n_k, h_k in zip(shape, steps):
        Ls.append(laplace_1d(n_k, h_k, dtype))
        Is.append(np.eye(n_k, dtype=dtype))

    cores: list[MatrixCore[T]] = []

    if D == 1:
        G = np.zeros((1, shape[0], shape[0], 1), dtype=dtype)
        G[0, :, :, 0] = Ls[0]
        return TTMatrix([G])

    # First core: [L1, I1]
    n1 = shape[0]
    G1 = np.zeros((1, n1, n1, 2), dtype=dtype)
    G1[0, :, :, 0] = Ls[0]
    G1[0, :, :, 1] = Is[0]
    cores.append(G1)

    # Middle cores: [[Ik, 0], [Lk, Ik]]
    for k in range(1, D - 1):
        nk = shape[k]
        Gk = np.zeros((2, nk, nk, 2), dtype=dtype)

        Gk[0, :, :, 0] = Is[k]
        Gk[1, :, :, 0] = Ls[k]
        Gk[1, :, :, 1] = Is[k]

        cores.append(Gk)

    # Last core: [[ID], [LD]]
    nD = shape[-1]
    GD = np.zeros((2, nD, nD, 1), dtype=dtype)
    GD[0, :, :, 0] = Is[-1]
    GD[1, :, :, 0] = Ls[-1]
    cores.append(GD)

    return TTMatrix(cores)
