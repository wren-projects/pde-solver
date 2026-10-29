"""Shared helpers for the :mod:`wren_ttd` benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from wren_ttd import TTD

type Float = np.float64

DTYPE = np.dtype(np.float64)

# Tensor grids for benchmarks that stay in the compressed TTD representation.
TT_DIMS = (3, 6, 10)
TT_MODES = (16, 128)  # , 1024, 4096
TT_RANKS = (2, 8)  # , 24, 64

# Tensor grids for benchmarks that need a dense array. The dense shape grows
# as mode ** dims, so only a much smaller set of modes is feasible.
DENSE_DIMS = TT_DIMS
DENSE_MODES = (4, 16, 64, 256, 512, 1024)
DENSE_RANKS = TT_RANKS

# A single dense tensor is capped at 2 GiB. Compression and expansion need
# several copies of the data (the dense array, the TTD cores, SVD workspace,
# and the result), which keeps the peak memory of each dense benchmark within
# roughly 8 GiB.
DENSE_MAX_TENSOR_BYTES = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class TensorCase:
    """Description of one benchmark tensor."""

    dims: int
    mode: int
    rank: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dense shape of the tensor."""
        return (self.mode,) * self.dims

    @property
    def dense_bytes(self) -> int:
        """Return the size of the dense representation in bytes."""
        return math.prod(self.shape) * DTYPE.itemsize

    @property
    def id(self) -> str:
        """Return a stable, readable pytest identifier."""
        return f"d{self.dims}-n{self.mode}-r{self.rank}"


def random_ttd(
    shape: tuple[int, ...],
    rank: int,
    *,
    seed: int = 0,
    dtype: np.dtype[Float] = DTYPE,
) -> TTD[Float]:
    """
    Create a TTD directly from random cores.

    Parameters
    ----------
    shape : tuple of ints
        The dense shape of the tensor.
    rank : int
        The internal TT-rank. All internal ranks are equal; the boundary
        ranks are 1.
    seed : int, optional
        Seed for the random number generator, by default 0.
    dtype : numpy.dtype, optional
        The dtype of the cores, by default ``numpy.float64``.

    Returns
    -------
    TTD
        A TTD with cores filled with standard normal random values.

    """
    rng = np.random.default_rng(seed)
    ranks = (1, *(rank for _ in range(len(shape) - 1)), 1)
    cores = [
        rng.standard_normal((ranks[i], n, ranks[i + 1]), dtype=dtype)
        for i, n in enumerate(shape)
    ]
    return TTD(cores)


TT_CASES = tuple(
    TensorCase(dims=dims, mode=mode, rank=rank)
    for dims in TT_DIMS
    for mode in TT_MODES
    for rank in TT_RANKS
)

DENSE_CASES = tuple(
    TensorCase(dims=dims, mode=mode, rank=rank)
    for dims in DENSE_DIMS
    for mode in DENSE_MODES
    for rank in DENSE_RANKS
    if math.prod((mode,) * dims) * DTYPE.itemsize <= DENSE_MAX_TENSOR_BYTES
)
