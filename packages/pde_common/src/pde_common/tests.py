import math

import numpy as np

from pde_common.types import NDArray

rng = np.random.default_rng(0)


type TestTensor = NDArray[np.float64]
type TestTensorPair = tuple[TestTensor, TestTensor]


def arange_tensor(*shape: int) -> TestTensor:
    """Create a tensor of given shape using np.arange."""
    return np.arange(1, math.prod(shape) + 1, dtype=np.float64).reshape(shape)


TEST_TENSORS: list[TestTensor] = [
    tensor.astype(np.float64)
    for tensor in [
        # ---- 2D ----
        arange_tensor(3, 3),
        rng.random((5, 3)),
        np.array([x % 2 == 0 for x in range(16)]).reshape(4, 4),
        # ---- 3D ----
        arange_tensor(4, 5, 5),
        rng.normal(size=(3, 4, 6)),
        # ---- 4D ----
        arange_tensor(6, 5, 4, 3),
        10e50 * np.ones((3, 4, 3, 5)),
        # ---- 5D ----
        arange_tensor(4, 3, 4, 5, 3),
        -10e-5 * arange_tensor(4, 3, 4, 5, 3),
        # ---- 6D ----
        arange_tensor(4, 3, 4, 3, 3, 6),
    ]
]


# NOTE: each pair of tensors must have the same shape
TEST_PAIR_TENSORS: list[TestTensorPair] = [
    (a.astype(np.float64), b.astype(np.float64))
    for a, b in [
        # ---- 2D ----
        (
            arange_tensor(3, 4),
            rng.integers(0, 10, size=(3, 4)),
        ),
        # ---- 3D ----
        (arange_tensor(2, 3, 4), np.ones((2, 3, 4))),
        # ---- 4D ----
        (
            arange_tensor(2, 3, 4, 5),
            rng.normal(size=(2, 3, 4, 5)),
        ),
        # ---- 5D ----
        (
            np.zeros((2, 2, 3, 4, 3)),
            arange_tensor(2, 2, 3, 4, 3),
        ),
        # ---- 6D ----
        (
            arange_tensor(2, 3, 2, 3, 2, 4),
            rng.integers(0, 5, size=(2, 3, 2, 3, 2, 4)),
        ),
        # ---- 7D ----
        (
            arange_tensor(2, 2, 3, 2, 4, 3, 2),
            np.ones((2, 2, 3, 2, 4, 3, 2)),
        ),
        # ---- 8D ----
        (
            arange_tensor(2, 3, 2, 4, 2, 3, 2, 3),
            rng.random((2, 3, 2, 4, 2, 3, 2, 3)),
        ),
    ]
]

for a, b in TEST_PAIR_TENSORS:
    assert a.shape == b.shape
    assert a.dtype == b.dtype


SMALL_TEST_SCALARS: tuple[float, ...] = (1, -1, math.pi)
TEST_SCALARS: tuple[float, ...] = (1, 2, 0.5, -1, 0, math.pi, -math.e, 1e30, -1e30)


def tensor_interior[DType: np.number](
    tensor: NDArray[DType], order: int = 1
) -> NDArray[DType]:
    """
    Return the interior of a given tensor as a 1D array.

    Parameters
    ----------
    tensor : NDArray
        The tensor of which the interior is to be taken.
    order : int, optional
        The number of elements to be removed from each direction in each dimension.

    Returns
    -------
    NDArray
        The one dimensional array containing all the elements of the interior in the
        input tensor.

    """
    mask = np.zeros_like(tensor, dtype=bool)
    interior_slices = tuple(slice(order, -order) for _ in range(tensor.ndim))
    mask[interior_slices] = True
    return tensor[mask]


def tensor_boundary[DType: np.number](
    tensor: NDArray[DType], order: int = 1
) -> NDArray[DType]:
    """
    Return the boundary of a given tensor as a 1D array.

    Parameters
    ----------
    tensor : NDArray
        The tensor of which the boundary is to be taken.
    order : int, optional
        The number of elements to be kept from each direction in each dimension.

    Returns
    -------
    NDArray
        The one dimensional array containing all the elements of boundary in the
        input tensor.

    """
    mask = np.ones_like(tensor, dtype=bool)
    interior_slices = tuple(slice(order, -order) for _ in range(tensor.ndim))
    mask[interior_slices] = False
    return tensor[mask]
