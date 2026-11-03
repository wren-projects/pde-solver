from __future__ import annotations

import math
from collections.abc import Iterable, Reversible, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, cast

import numpy as np
from wren_common.types import Matrix, NDArray, Scalar, Vector

from wren_ttd.math import DEFAULT_EPSILON, qr_rows
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def fold_right[DType: np.floating](array: NDArray[DType]) -> Matrix[DType]:
    """
    Flatten all but the first dimension of an array.

    Parameters
    ----------
    array : NDArray[DType]
        The array to flatten in shape ``(α, β₂, β₃, …)``.

    Returns
    -------
    Matrix[DType]
        The flattened array in shape ``(α, β₂ ⋅ β₃ ⋅ …)``.

    """
    return array.reshape(array.shape[0], -1)


def unfold_right[DType: np.floating, Shape: tuple[int, ...]](
    matrix: Matrix[DType], original_shape: Shape
) -> np.ndarray[Shape, np.dtype[DType]]:
    """
    Restore all but the first dimension to the original shape of a right-folded array.

    This is the inverse of :func:`fold_right`.

    Parameters
    ----------
    matrix : Matrix[DType]
        The flattened array in shape ``(α₁, β₂ ⋅ β₃ ⋅ …)``.
    original_shape : Shape
        The original shape of the array ``(β₁, β₂, β₃, …)``.

    Returns
    -------
    np.ndarray[Shape, np.dtype[DType]]
        The unfolded array in the shape ``(α₁, β₂, β₃, …)``.

    """
    shape = (-1, *original_shape[1:])
    return cast(np.ndarray[Shape, np.dtype[DType]], matrix.reshape(shape))


def fold_left[DType: np.floating](array: NDArray[DType]) -> Matrix[DType]:
    """
    Flatten all but the last dimension of an array.

    Parameters
    ----------
    array : NDArray[DType]
        The array to flatten in shape ``(β₁, β₂, …, α)``.

    Returns
    -------
    Matrix[DType]
        The flattened array in shape ``(β₁ ⋅ β₂ ⋅ …, α)``.

    """
    return array.reshape(-1, array.shape[-1])


def unfold_left[DType: np.floating, Shape: tuple[int, ...]](
    matrix: Matrix[DType], original_shape: Shape
) -> np.ndarray[Shape, np.dtype[DType]]:
    """
    Restore all but the last dimension to the original shape of a left-folded array.

    This is the inverse of :func:`fold_left`.

    Parameters
    ----------
    matrix : Matrix[DType]
        The flattened array in shape ``(β₁ ⋅ β₂ ⋅ …, α)``.
    original_shape : Shape
        The original shape of the array ``(β₁, β₂, …, βₙ)``.

    Returns
    -------
    np.ndarray[Shape, np.dtype[DType]]
        The unfolded array in the shape ``(β₁, β₂, …, α)``.

    """
    shape = (*original_shape[:-1], -1)
    return cast(np.ndarray[Shape, np.dtype[DType]], matrix.reshape(shape))


def apply_left_message[DType: np.floating](
    message: Matrix[DType],
    core: Core[DType],
    *,
    buffer: Vector[DType] | None = None,
) -> Core[DType]:
    """
    Contract a message matrix into the left rank of a core.

    Computes ``𝐌 ⋅ 𝐆``, i.e. the mode-1 product ``𝐆 ×₁ 𝐌``, where ``𝐆`` is a
    core ``(α, i, β)`` and ``𝐌`` is a matrix (``(m, α)``) whose number of
    columns matches ``α``.

    Parameters
    ----------
    message : Matrix[DType]
        The message matrix.
    core : Core[DType]
        The core to apply the message to.
    buffer : Vector[DType] | None, optional
        Buffer of shape ``m ⋅ i ⋅ β``, to write the result into, by default
        None (allocate a new array).

    Returns
    -------
    Core[DType]
        The contracted core of shape ``(m, i, β)``.

    """
    left, mode, right = core.shape
    msg_rows, msg_cols = message.shape
    assert msg_cols == left, "core and message must be compatible"

    if buffer is not None:
        columns = mode * right
        assert buffer.ndim == 1, "Buffer should be a 1D array"
        if buffer.size < msg_rows * columns:
            raise ValueError(
                f"Buffer must be at least {msg_rows * columns} elements long"
            )
        matrix_buffer = buffer[: msg_rows * columns].reshape(msg_rows, columns)
    else:
        matrix_buffer = np.empty((msg_rows, mode * right), dtype=core.dtype)

    return np.matmul(message, fold_right(core), out=matrix_buffer).reshape(
        msg_rows, mode, right
    )


def apply_right_message[DType: np.floating](
    core: Core[DType],
    message: Matrix[DType],
    *,
    buffer: Vector[DType] | None = None,
) -> Core[DType]:
    """
    Contract a message matrix into the right rank of a core.

    Computes ``𝐆 ⋅ 𝐌``, i.e. the mode-3 product ``𝐆 ×₃ 𝐌``, where ``𝐆`` is a
    core ``(α, i, β)`` and ``𝐌`` is a matrix (``(β, n)``) whose number of rows
    matches ``β``.

    Parameters
    ----------
    core : Core[DType]
        The core to apply the message to.
    message : Matrix[DType]
        The message matrix.
    buffer : Vector[DType] | None, optional
        Buffer of size ``α ⋅ i ⋅ n``, to write the result into, by default
        None (allocate a new array).

    Returns
    -------
    Core[DType]
        The contracted core of shape ``(α, i, n)``.

    """
    left, mode, right = core.shape
    msg_rows, msg_cols = message.shape
    assert msg_rows == right, "core and message must be compatible"

    if buffer is not None:
        rows = left * mode
        assert buffer.ndim == 1, "Buffer should be a 1D array"
        if buffer.size < rows * msg_cols:
            raise ValueError(f"Buffer must be at least {rows * msg_cols} elements long")
        matrix_buffer = buffer[: rows * msg_cols].reshape((rows, msg_cols))
    else:
        matrix_buffer = np.empty((left * mode, msg_cols), dtype=core.dtype)

    return np.matmul(fold_left(core), message, out=matrix_buffer).reshape(
        left, mode, msg_cols
    )


def contract_pair[DType: np.floating](
    a: Core[DType],
    b: Core[DType],
) -> Matrix[DType]:
    """
    Contract the right rank of one core with the left rank of another.

    Computes ``𝐀 ⋅ 𝐁`` over the shared rank, where ``𝐀`` is ``(α, i, r)`` and
    ``𝐁`` is ``(r, j, β)``. Equivalent to :func:`numpy.tensordot` with
    ``axes=1``, but uses :func:`numpy.matmul` to go through BLAS directly.

    Parameters
    ----------
    a : Core[DType]
        The left core.
    b : Core[DType]
        The right core.

    Returns
    -------
    Matrix[DType]
        The contracted matrix of shape ``(α ⋅ i, j ⋅ β)``.

    """
    _, _, rank = a.shape
    assert rank == b.shape[0], "cores must have matching ranks"
    return cast(
        Matrix[DType],
        np.matmul(a.reshape(-1, rank), b.reshape(rank, -1)),
    )


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
    # Reuse a single buffer for the QR propagation. Each result is consumed by
    # the next QR decomposition before the buffer is overwritten.
    max_core_size = max(core.size for core in cores)
    buffer = np.empty(max_core_size, dtype=cores[0].dtype)

    for k in range(len(cores), 1, -1):  # for k = d to 2 step -1
        # [𝐆ₖ(βₖ₋₁; iₖβₖ), R(αₖ₋₁, βₖ₋₁)] := QR_rows(𝐆ₖ(αₖ₋₁; iₖβₖ))
        # G = 𝐆ₖ(αₖ₋₁; iₖβₖ)
        core = cores[k - 1]
        # 𝐐, 𝐑 = QR_rows(𝐆ₖ(αₖ₋₁; iₖβₖ))
        q, r = qr_rows(fold_right(core))
        # 𝐆ₖ(βₖ₋₁; iₖβₖ) = 𝐐
        cores[k - 1] = unfold_right(q, core.shape)
        # 𝐆ₖ₋₁ := 𝐆ₖ₋₁ ×₃ 𝐑
        cores[k - 2] = apply_right_message(cores[k - 2], r, buffer=buffer)


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
    scratch_buffer: Vector[DType] = np.empty(max_brank * max_alnrank, dtype=dtype)
    result_buffer: Matrix[DType] = np.empty((max_brank, max_arank), dtype=dtype)

    # result = 1 × 1
    result = np.ones((1, 1), dtype=dtype)

    # ar × n × al, br × n × bl
    for a_core, b_core in zip(a_cores, b_cores, strict=True):
        _, _, al = a_core.shape
        _, _, bl = b_core.shape
        # br × ar, ar × n × al -> br × n × al
        result = apply_left_message(result, a_core, buffer=scratch_buffer)

        result = np.matmul(
            # br × n × bl -> (br × n) × bl -> bl × (br × n)
            fold_left(b_core).T,
            # br × n × al -> (br × n) × al
            fold_left(result),
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
