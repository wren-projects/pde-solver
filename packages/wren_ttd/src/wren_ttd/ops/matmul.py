# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from wren_common.types import NDArray

from wren_ttd._helpers import multiply_cores
from wren_ttd._numpy_api import implements_ufunc

if TYPE_CHECKING:
    from wren_ttd.core import TTD

from .tensordot import tensordot
from .transpose import transpose

MATRIX_NDIM = 2


def _contraction_b_axis(b: TTD[Any]) -> int:
    return 0 if b.ndim == 1 else b.ndim - MATRIX_NDIM


@implements_ufunc("matmul")
def matmul[DType: np.floating](
    a: TTD[DType],
    b: TTD[DType],
    *,
    out: TTD[DType] | None = None,
) -> TTD[DType] | DType:
    """
    Compute the matrix product of two TTDs.

    This operation follows the vector and matrix dimension conventions of
    :func:`numpy.matmul`: a 1D left operand is treated as a row vector, and a
    1D right operand as a column vector. The added dimensions are removed from
    the result, so multiplying two vectors returns a scalar. If any of the
    operands has more than two dimensions, a batched matrix multiplication is
    calculated. See :func:`numpy.matmul` for more details.

    Both operands must be TTDs with the same dtype.

    Parameters
    ----------
    a : TTD[DType]
        Left operand. Its last dimension is contracted with the corresponding
        matrix dimension of `b`.
    b : TTD[DType]
        Right operand. If 1D, its first dimension is contracted; otherwise
        its second-to-last dimension is contracted. Must have the same dtype
        as `a`.
    out : TTD[DType], optional
        TTD to receive the result in a direct call to :func:`matmul`. It must
        have the result's shape and dtype, and is supported only when the
        result is a TTD. The ``out`` parameter of the NumPy ufunc interface is
        not supported.

    Returns
    -------
    TTD[DType] | DType
        The matrix product as a TTD, or a NumPy scalar when both operands are
        vectors.

    Raises
    ------
    NotImplementedError
        If both operands have leading batch dimensions.
    TypeError
        If `out` is provided for a scalar result. A NumPy ufunc call with a
        non-TTD operand is also unsupported and may raise this error.
    ValueError
        If the contracted dimensions, operand dtypes, or `out` shape/dtype do
        not match, or if two batch shapes are not broadcast-compatible.

    Notes
    -----
    NumPy's `dtype`, `casting`, `order`, `subok`, `signature`, and `out`
    options are not supported. A NumPy call with a dense/TTD mixed pair is not
    handled by this implementation. The direct `ops.matmul` function accepts
    `out` as a TTD.

    See Also
    --------
    numpy.matmul : Equivalent operation for dense arrays.
    tensordot : Contract arbitrary pairs of tensor axes.
    inner_product : Compute the inner product of two TTDs.
    TTD.__matmul__ : Matrix multiplication with the ``@`` operator.

    Examples
    --------
    >>> import numpy as np
    >>> from wren_ttd import ops
    >>> from wren_ttd.core import TTD
    >>> a = TTD(np.arange(6.0).reshape(2, 3))
    >>> b = TTD(np.arange(12.0).reshape(3, 4))
    >>> ops.matmul(a, b).shape
    (2, 4)
    >>> vector = TTD([1.0, 2.0, 3.0])
    >>> ops.matmul(vector, vector)
    np.float64(14.0)

    """
    from wren_ttd.core import TTD

    if a.dtype != b.dtype:
        raise ValueError("TTD objects must have the same dtype")

    _validate_batch_dimensions(a, b)
    result = _compute_matmul(a, b)

    if out is None:
        return result

    if not isinstance(result, TTD):
        raise TypeError("out is only supported for TTD-valued results")

    if out.shape != result.shape:
        raise ValueError("Output tensor has an incorrect shape.")
    if out.dtype != result.dtype:
        raise ValueError("Output tensor has an incorrect dtype.")

    out.data = list(result.data)
    return out


def _validate_batch_dimensions[DType: np.floating](
    a: TTD[DType], b: TTD[DType]
) -> None:
    """Reject two-sided batched products after validating their batch shapes."""
    if a.shape[-1] != b.shape[_contraction_b_axis(b)]:
        raise ValueError("Input operand dimensions do not match for matmul")

    if a.ndim > MATRIX_NDIM and b.ndim > MATRIX_NDIM:
        try:
            _ = np.broadcast_shapes(a.shape[:-MATRIX_NDIM], b.shape[:-MATRIX_NDIM])
        except ValueError as error:
            raise ValueError("Batch dimensions are not broadcastable") from error


def _compute_matmul[DType: np.floating](
    a: TTD[DType], b: TTD[DType]
) -> TTD[DType] | DType:
    """Compute a supported product through a single TTD contraction."""
    from wren_ttd.core import TTD

    # none is batched
    if a.ndim <= MATRIX_NDIM and b.ndim <= MATRIX_NDIM:
        return tensordot(a, b, axes=1)

    # both are batched
    if a.ndim > MATRIX_NDIM and b.ndim > MATRIX_NDIM:
        # multiply the batch cores
        new_cores = multiply_cores(a.data[:-2], b.data[:-2])

        a_2, a_1 = a.data[-2:]
        b_2, b_1 = b.data[-2:]

        # contract the last core in `a` and the second the last in `b` – the
        # actual matrix multiplication – and merge the result into the second
        # to last core of `a`
        new_cores.append(
            cast(
                NDArray[DType], np.einsum("afc,cde,bdg", a_2, a_1, b_2, optimize=True)
            ).reshape(-1, a_2.shape[1], b_2.shape[2])
        )
        # append last core of `b`
        new_cores.append(b_1)

        return TTD(new_cores)

    # at most one is batched
    b_axis = _contraction_b_axis(b)
    result = tensordot(a, b, axes=((-1,), (b_axis,)))

    assert isinstance(result, TTD), "batched matrix multiplication must return a TTD"

    if a.ndim == MATRIX_NDIM:
        return transpose(
            result,
            (*range(1, b.ndim - 1), 0, b.ndim - 1),
        )

    return result
