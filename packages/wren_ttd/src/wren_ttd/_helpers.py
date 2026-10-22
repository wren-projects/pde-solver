from __future__ import annotations
from wren_ttd._svd import SVD, delta_truncated_svd

import math
from collections.abc import Iterable, Reversible
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast, ParamSpec, Callable, Protocol

import numpy as np
from wren_common.math import dot_product, scale_matrix
from wren_common.types import Matrix, NDArray, Vector, Scalar

from wren_ttd.math import DEFAULT_EPSILON, qr_rows
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


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
    a_cores: Iterable[Core[DType]],
    b_cores: Iterable[Core[DType]],
    n: int,
) -> Matrix[DType]:
    """
    Contract two matching core sequences pairwise into a single matrix.

    Sums over the shared physical modes and all inner ranks, keeping only
    the outer rank of the first `a` core and the last `b` core.

    Parameters
    ----------
    a_cores : Iterable[Core[DType]]
        The cores of the first tensor.
    b_cores : Iterable[Core[DType]]
        The cores of the second tensor.
    n : int
        The number of cores in each sequence.

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


def ttsvd[DT: np.floating, **Params](
    array: NDArray[DT],
    svd_function: SVD[DT, Params],
    epsilon: np.floating | float = DEFAULT_EPSILON,
    *svd_args: Params.args,
    **svd_kwargs: Params.kwargs,
) -> list[Core[DT]]:
    d = array.ndim

    if d == 1:
        return [array.reshape((1, len(array), 1))]

    delta = truncation_parameter(array, epsilon)

    # 𝐂 = reshape(𝐂, [n, -1])
    residue: Matrix[DT] = array.reshape((array.shape[0], -1))

    # r₀ = 1
    r = 1

    cores: list[Core[DT]] = []

    # for k = 1 to d - 1 do
    # note: n = nₖ, r = rₖ₋₁
    for n in array.shape[:-1]:
        # 𝐂 = reshape(𝐂, [rₖ₋₁ nₖ, -1])
        residue = residue.reshape((r * n, -1))

        # 𝐔, 𝐒, 𝐕ᵀ = SVDᵟ(𝐂, δ)
        u, s, v_t = svd_function(residue, delta, *svd_args, **svd_kwargs)

        # 𝐆ₖ = reshape(U, [rₖ₋₁, nₖ, rₖ])
        new_core = u.reshape((r, n, -1))
        cores.append(new_core)

        # set rₖ = rankᵟ(C) for next iteration
        r = new_core.shape[2]

        # 𝐂 = 𝐒𝐕ᵀ
        residue = scale_matrix(s, v_t)

    cores.append(residue.reshape((*residue.shape, 1)))

    return cores
