from __future__ import annotations

import math
from collections.abc import Iterable, Reversible
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from wren_common.math import dot_product
from wren_common.types import Matrix, NDArray

from wren_ttd.math import DEFAULT_EPSILON, qr_rows
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def reverse_cores[DType: np.floating](
    cores: Reversible[Core[DType]],
) -> Iterable[Core[DType]]:
    return (core.T for core in reversed(cores))


def to_int_tuple(value: int | Iterable[int]) -> tuple[int, ...]:
    """
    Convert an int or iterable of ints to a tuple of ints.

    If the input is an int, it is converted to a tuple of length 1. An iterable
    is converted to a tuple of the same length. Useful for normalizing
    NumPy-style shape or axis arguments to single type.

    See also: :func:`numpy.core.multiarray.normalize_axis_tuple` if the handling
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
    a_cores: Iterable[Core[DType]],
    b_cores: Iterable[Core[DType]],
    n: int,
) -> Matrix[DType]:
    """
    Contract pairwise the lists of cores A and B using repeated tensordot.

    Results in a matrix of shape .

    Parameters
    ----------
    a_cores : Iterable[Core[DType]]
        The cores of the first tensor.
    b_cores : Iterable[Core[DType]]
        The cores of the second tensor.
    n : int
        The number of cores in each tensor.

    Returns
    -------
    Matrix[DType]
        The resulting matrix.

    """
    # ‌The generated einsum expression is in the form ABC,GBI,CDE,IDK->AEGK. ABC
    # is the first core of A, GBI is the first core of B, CDE is the second core
    # of A, …. Consequently, it first sums the matching cores along the second
    # (rank) axis (ABC,GBI->ACGI / CDE,IDK->CEIK), then sums the results along
    # all axes except the first and the last of a and b each (ACGI,CEIK->AEGK).

    # NOTE: See `TTD.__array__` for the general generated einsum index idea.

    summation_indices: list[Any] = [
        item
        for i, (a_core, b_core) in enumerate(zip(a_cores, b_cores, strict=True))
        for item in [
            a_core,
            (2 * i + 0, 2 * i + 1, 2 * i + 2),
            b_core,
            (2 * (n + i + 1) + 0, 2 * i + 1, 2 * (n + i + 1) + 2),
        ]
    ]

    result = cast(
        Matrix[DType],
        np.einsum(*summation_indices, optimize=True),  # pyright: ignore[reportAny]
    )

    # assumes, that both sequences started from the boundary core -> |A| = |G| = 1
    return result.squeeze((0, 2))


def truncation_parameter[DT: np.floating](
    tensor: NDArray[DT] | TTD[DT], epsilon: np.floating | float = DEFAULT_EPSILON
) -> DT:
    """
    Compute the truncation parameter of a tensor.

    It uses the formula δ = (ε / √(d - 1)) ⋅ ‖A‖ᶠ.

    Parameters
    ----------
    tensor : NDArray[DT]
        The tensor to compute the truncation parameter of.
    epsilon : np.floating | float, optional
        The error tolerance for the compression, by default DEFAULT_EPSILON.

    Returns
    -------
    DT
        The truncation parameter.

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
    Stack cores into a single block core.

    For cores C₀, C₁, ..., Cₙ with shapes (lᵢ, n, rᵢ), the block core is
    defined as

        ⌈ C₀ 0  … 0 ⌉
        | 0  C₁ … 0 |
        | ⋮  ⋮  ⋱ ⋮ |
        ⌊ 0  0  … Cₙ⌋

    with shape (l₀ + ⋯ + lₙ, n, r₀ + ⋯ + rₙ).

    Parameters
    ----------
    blocks : tuple[Core[DType], ...]
        The cores to stack.

    Returns
    -------
    Core[DType]
        The block core.

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
