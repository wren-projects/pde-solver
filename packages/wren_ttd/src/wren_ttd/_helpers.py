from __future__ import annotations

import math
from collections.abc import Iterable, Reversible, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any, SupportsIndex, cast

import numpy as np
import numpy.typing as npt
from wren_common.types import Matrix, NDArray, Scalar, Vector

from wren_ttd.math import DEFAULT_EPSILON, qr_rows
from wren_ttd.types import Core

if TYPE_CHECKING:
    from wren_ttd.core import TTD


def normalize_dtype(dtype: npt.DTypeLike | None) -> np.dtype[Any]:
    """Normalize and validate a floating-point TTD dtype."""
    normalized = np.dtype(dtype)
    if not np.issubdtype(normalized, np.floating):
        raise TypeError(f"{normalized!r} is not a valid dtype for TTD")
    return normalized


def promote_dtype(*dtypes: npt.DTypeLike | Scalar) -> np.dtype[Any]:
    """
    Determine the result dtype of an operation from its operand dtypes.

    Uses :func:`numpy.result_type`, i.e. NumPy 2 (NEP 50) promotion rules:
    NumPy scalars and arrays are strong, while Python scalars are weak and
    adopt the dtype of the other operand.

    Parameters
    ----------
    *dtypes : npt.DTypeLike | Scalar
        The operand dtypes or scalar values to promote.

    Returns
    -------
    np.dtype
        The promoted floating-point dtype.

    Raises
    ------
    TypeError
        If the promoted dtype is not a floating-point dtype.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd._helpers import promote_dtype
    >>> promote_dtype(np.dtype(np.float32), np.dtype(np.float64))
    dtype('float64')
    >>> promote_dtype(np.dtype(np.float32), 1.0)
    dtype('float32')
    >>> promote_dtype(np.dtype(np.float32), np.float64(1))
    dtype('float64')

    """
    return normalize_dtype(np.result_type(*dtypes))


def validate_out[DType: np.floating](
    out: TTD[DType],
    shape: tuple[int, ...],
    result_dtype: np.dtype[Any],
) -> np.dtype[DType]:
    """
    Validate an explicit output TTD and return the dtype to cast the result to.

    The result may be cast to the dtype of `out` with NumPy's ``"same_kind"``
    casting rules, so in-place operations can narrow floating-point results.

    Parameters
    ----------
    out : TTD[DType]
        The output TTD to validate.
    shape : tuple[int, ...]
        The expected shape of the result.
    result_dtype : np.dtype
        The dtype of the computed result.

    Returns
    -------
    np.dtype[DType]
        The dtype of `out`, i.e. the dtype the result cores have to be cast to.

    Raises
    ------
    ValueError
        If `out` has a different shape or its dtype cannot represent the
        result under ``"same_kind"`` casting.

    """
    if out.shape != shape:
        raise ValueError(f"Output tensor has shape {out.shape}, expected {shape}")

    if not np.can_cast(result_dtype, out.dtype, casting="same_kind"):
        raise ValueError(
            f"Cannot cast the result from {result_dtype} to the output dtype "
            f"{out.dtype} with casting='same_kind'"
        )

    return out.dtype


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


def mode_1_product[DType: np.floating](
    matrix: Matrix[DType],
    core: Core[DType],
    *,
    buffer: Vector[DType] | None = None,
) -> Core[DType]:
    """
    Contract a matrix into the left rank of a core.

    Computes ``𝐆 ×₁ 𝐌``, where ``𝐆`` is a core ``(α, i, β)`` and ``𝐌`` is a
    matrix (``(m, α)``).

    Parameters
    ----------
    matrix : Matrix[DType]
        The matrix.
    core : Core[DType]
        The core to apply the matrix to.
    buffer : Vector[DType] | None, optional
        Buffer of size ``m ⋅ i ⋅ β``, to write the result into, by default
        None (allocate a new array).

    Returns
    -------
    Core[DType]
        The contracted core of shape ``(m, i, β)``.

    """
    left, mode, right = core.shape
    mat_rows, mat_cols = matrix.shape
    assert mat_cols == left, "core and matrix must be compatible"

    if buffer is not None:
        columns = mode * right
        size = mat_rows * columns
        assert buffer.ndim == 1, "Buffer should be a 1D array"
        if buffer.size < size:
            raise ValueError(f"Buffer must be at least {size} elements long")
        matrix_buffer = buffer[:size].reshape(mat_rows, columns)
    else:
        matrix_buffer = None

    return np.matmul(matrix, fold_right(core), out=matrix_buffer).reshape(
        mat_rows, mode, right
    )


def mode_3_product[DType: np.floating](
    core: Core[DType],
    matrix: Matrix[DType],
    *,
    buffer: Vector[DType] | None = None,
) -> Core[DType]:
    """
    Contract a matrix into the right rank of a core.

    Computes ``𝐆 ×₃ 𝐌``, where ``𝐆`` is a core ``(α, i, β)`` and ``𝐌`` is a
    matrix (``(β, n)``).

    Parameters
    ----------
    core : Core[DType]
        The core to apply the matrix to.
    matrix : Matrix[DType]
        The matrix.
    buffer : Vector[DType] | None, optional
        Buffer of size ``α ⋅ i ⋅ n``, to write the result into, by default
        None (allocate a new array).

    Returns
    -------
    Core[DType]
        The contracted core of shape ``(α, i, n)``.

    """
    left, mode, right = core.shape
    mat_rows, mat_cols = matrix.shape
    assert right == mat_rows, "core and matrix must be compatible"

    if buffer is not None:
        rows = left * mode
        size = rows * mat_cols
        assert buffer.ndim == 1, "Buffer should be a 1D array"
        if buffer.size < size:
            raise ValueError(f"Buffer must be at least {size} elements long")
        matrix_buffer = buffer[:size].reshape(rows, mat_cols)
    else:
        matrix_buffer = None

    return np.matmul(fold_left(core), matrix, out=matrix_buffer).reshape(
        left, mode, mat_cols
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
        np.matmul(fold_left(a), fold_right(b)),
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


def to_int_tuple(value: SupportsIndex | Iterable[SupportsIndex]) -> tuple[int, ...]:
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
    return (int(value),) if isinstance(value, SupportsIndex) else tuple(map(int, value))


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
        cores[k - 2] = mode_3_product(cores[k - 2], r, buffer=buffer)


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
        result = mode_1_product(result, a_core, buffer=scratch_buffer)

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


def merge_cores[DType: np.floating](
    left: Core[DType],
    right: Core[DType],
) -> Core[DType]:
    """
    Merge the modes of two adjacent cores into a single mode.

    Computes the core ``C`` of shape ``(α, i ⋅ j, β)`` with

    .. math::

        C(α; i j; β) = ∑_r A(α; i; r) B(r; j; β),

    where ``A`` is ``(α, i, r)`` and ``B`` is ``(r, j, β)``. This is exact and
    does not change the tensor described by the cores.

    Parameters
    ----------
    left : Core[DType]
        The left core of shape ``(α, i, r)``.
    right : Core[DType]
        The right core of shape ``(r, j, β)``.

    Returns
    -------
    Core[DType]
        The merged core of shape ``(α, i ⋅ j, β)``.

    See Also
    --------
    split_core : Inverse operation, which may inflate the ranks.
    contract_pair : Contract the shared rank of two cores.

    """
    left_rank, left_mode, _ = left.shape
    _, right_mode, right_rank = right.shape
    return contract_pair(left, right).reshape(
        left_rank, left_mode * right_mode, right_rank
    )


def split_core[DType: np.floating](
    core: Core[DType],
    left_mode: int,
) -> tuple[Core[DType], Core[DType]]:
    """
    Split one mode of a core into two adjacent modes.

    The mode ``n`` of `core` is factored as ``n = i ⋅ j`` with ``i`` equal to
    `left_mode`. The two resulting cores describe the same tensor, but the
    split introduces a new TT rank bounded by ``min(α ⋅ i, j ⋅ β)``.

    Parameters
    ----------
    core : Core[DType]
        The core of shape ``(α, n, β)`` to split.
    left_mode : int
        The size ``i`` of the first resulting mode. It must divide the mode
        size ``n``.

    Returns
    -------
    tuple[Core[DType], Core[DType]]
        The two cores of shapes ``(α, i, k)`` and ``(k, j, β)``.

    See Also
    --------
    merge_cores : Inverse operation, which is exact and rank-preserving.

    """
    left_rank, mode, right_rank = core.shape
    right_mode = mode // left_mode

    u, s, v_t = np.linalg.svd(
        core.reshape(left_rank * left_mode, right_mode * right_rank),
        full_matrices=False,
    )

    return (
        cast(Core[DType], (u * s).reshape(left_rank, left_mode, -1)),
        cast(Core[DType], v_t.reshape(-1, right_mode, right_rank)),
    )


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


def singleton_cores[DType: np.floating](
    shape: tuple[int, ...], dtype: np.dtype[DType]
) -> list[Core[DType]]:
    return [np.ones((1, n, 1), dtype=dtype) for n in shape]
