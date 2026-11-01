from __future__ import annotations

import math
from collections.abc import Iterable, Reversible, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, cast

import numpy as np
from wren_common.math import dot_product
from wren_common.types import Matrix, NDArray, Scalar

from wren_ttd.math import DEFAULT_EPSILON, qr_rows
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def smallest_core[DType: np.floating](
    cores: Iterable[Core[DType]],
) -> tuple[Core[DType], int]:
    """
    Find the smallest core by size and its index.

    Parameters
    ----------
    cores : Iterable[Core]
        The cores to find the smallest one of.

    Returns
    -------
    tuple[Core, int]
        The smallest core and its index.

    """
    _, index, core = min((core.size, index, core) for index, core in enumerate(cores))
    return core, index


def reverse_cores[DType: np.floating](
    cores: Reversible[Core[DType]],
) -> Iterable[Core[DType]]:
    """
    Reverse the order of the cores and transpose each one.

    Implementation of :func:`numpy.ndarray.T` for TTDs.

    Parameters
    ----------
    cores : Reversible[Core[DType]]
        The cores to reverse.

    Returns
    -------
    Iterable[Core[DType]]
        The reversed cores.

    """
    return (core.T for core in reversed(cores))


def to_int_tuple(value: int | Iterable[int]) -> tuple[int, ...]:
    """
    Normalize an int or iterable of ints to a tuple of ints.

    See also :func:`numpy.lib.array_utils.normalize_axis_tuple` when handling
    of negative and out-of-bounds indices is desired.

    Parameters
    ----------
    value : int | Iterable[int]
        The input to convert.

    Returns
    -------
    tuple[int, ...]
        The converted tuple.

    """
    return (int(value),) if isinstance(value, int) else tuple(map(int, value))


def orthogonalize_right[DType: np.floating](cores: list[Core[DType]]) -> None:
    """Orthogonalize the cores in place using a right-to-left QR sweep."""
    for k in range(len(cores), 1, -1):  # for k = d to 2 step -1
        # [𝐆ₖ(βₖ₋₁; iₖβₖ), R(αₖ₋₁, βₖ₋₁)] := QR_rows(𝐆ₖ(αₖ₋₁; iₖβₖ))
        # G = 𝐆ₖ(αₖ₋₁; iₖβₖ)
        core = cores[k - 1]
        alpha_k1, i_k, beta_k = core.shape
        # 𝐐, 𝐑 = QR_rows(𝐆ₖ(αₖ₋₁; iₖβₖ)) = QR(𝐆ₖ(αₖ₋₁; iₖβₖ)ᵀ)ᵀ
        q, r = qr_rows(core.reshape((alpha_k1, i_k * beta_k)))
        # 𝐆ₖ(βₖ₋₁; iₖβₖ) = 𝐐
        cores[k - 1] = q.reshape((-1, i_k, beta_k))
        # 𝐆ₖ₋₁ := 𝐆ₖ₋₁ ×₃ 𝐑
        # NOTE: there is a typo in the TTD paper: it incorrectly says 𝐆ₖ ×₃ 𝐑
        cores[k - 2] = dot_product(cores[k - 2], r)


def contract_cores[DType: np.floating](
    a_cores: Sequence[Core[DType]],
    b_cores: Sequence[Core[DType]],
    dtype: np.dtype[DType],
) -> Matrix[DType]:
    """
    Contract two matching core sequences pairwise into a single matrix.

    Sums over the shared physical modes and all inner ranks, keeping only
    the outer rank of the first `a` core and the last `b` core.

    Parameters
    ----------
    a_cores : Sequence[Core[DType]]
        The cores of the first tensor.
    b_cores : Sequence[Core[DType]]
        The cores of the second tensor.
    dtype : np.dtype[DType]
        The dtype of the resulting matrix. Should match the dtype of the cores.

    Returns
    -------
    Matrix[DType]
        The resulting matrix.

    """
    # Determine the maximum needed buffer sizes
    max_arank, max_brank = (
        max(core.shape[axis] for core in cores for axis in (0, 2))
        for cores in (a_cores, b_cores)
    )
    max_alnrank = max(core.shape[1] * core.shape[2] for core in a_cores)

    # Allocate buffers so that they can be reused in all iterations
    scratch_buffer = np.empty((max_brank, max_alnrank), dtype=dtype)
    result_buffer = np.empty((max_brank, max_arank), dtype=dtype)

    # result = 1 × 1
    result = np.ones((1, 1), dtype=dtype)

    for a_core, b_core in zip(a_cores, b_cores, strict=True):
        ar, n, al = a_core.shape
        br, _, bl = b_core.shape
        result = np.matmul(
            # br × ar
            result,
            # ar × n × al -> ar × (n × al)
            a_core.reshape(ar, -1),
            # -> br × (n × al)
            out=scratch_buffer[:br, : n * al],
        )
        result = np.matmul(
            # br × n × bl -> (br × n) × bl -> bl × (br × n)
            b_core.reshape(-1, bl).T,
            # br × (n × al) -> (br × n) × al
            result.reshape(-1, al),
            # -> bl × al
            out=result_buffer[:bl, :al],
        )

    return result.T


def truncation_parameter[DT: np.floating](
    tensor: NDArray[DT] | TTD[DT], epsilon: Scalar = DEFAULT_EPSILON
) -> DT:
    """
    Compute the per-SVD truncation tolerance δ = (ε / √(d - 1)) ⋅ ‖A‖ᶠ.

    Parameters
    ----------
    tensor : NDArray[DT] | TTD[DT]
        The tensor to compute the tolerance of.
    epsilon : np.floating | float, optional
        The target relative error, by default DEFAULT_EPSILON.

    Returns
    -------
    DT
        The truncation tolerance.

    Raises
    ------
    ValueError
        If the tensor has less than 2 dimensions.

    """
    d = tensor.ndim
    if d <= 1:
        raise ValueError("Tensor must be at least 2D")

    return tensor.dtype.type(epsilon / math.sqrt(d - 1)) * np.linalg.norm(tensor)


def block_core[DType: np.floating](
    blocks: tuple[Core[DType], ...],
) -> Core[DType]:
    """
    Arrange cores of shape (lᵢ, n, rᵢ) into a block-diagonal core.

    The result has shape (l₀ + ⋯ + lₙ, n, r₀ + ⋯ + rₙ).

    Parameters
    ----------
    blocks : tuple[Core[DType], ...]
        The cores to stack. All must share the same mode size `n`.

    Returns
    -------
    Core[DType]
        The block-diagonal core.

    """
    n = blocks[0].shape[1]
    dtype = blocks[0].dtype

    # compute offsets of the blocks
    l_offsets, r_offsets = (
        cast(
            list[int],  # not really, but it's to satisfy type checkers
            # cumsum of left/right ranks
            np.r_[0, np.cumsum([c.shape[axis] for c in blocks])],
        )
        for axis in (0, 2)
    )

    G = np.zeros((l_offsets[-1], n, r_offsets[-1]), dtype=dtype)

    for core, (l0, l1), (r0, r1) in zip(
        blocks, pairwise(l_offsets), pairwise(r_offsets), strict=True
    ):
        G[l0:l1, :, r0:r1] = core

    return G
