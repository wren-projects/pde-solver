import math

import numpy as np

from numpy_ttd import DEFAULT_EPSILON, TTD
from numpy_ttd.types import Scalar
from pde_solver.pde_types import NDArray

rng = np.random.default_rng(0)


type TestTensor = NDArray
type TestTensorPair = tuple[TestTensor, TestTensor]
type TestTTD = TTD[np.float64]
type TestTTDPair = tuple[TTD[np.float64], TTD[np.float64]]


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

TEST_TTD: list[tuple[TestTensor, TestTTD]] = [
    (tensor, TTD.from_ndarray(tensor)) for tensor in TEST_TENSORS
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

TEST_PAIR_TTD: list[tuple[TestTensorPair, TestTTDPair]] = [
    ((a, b), (TTD.from_ndarray(a), TTD.from_ndarray(b))) for a, b in TEST_PAIR_TENSORS
]

TEST_SHAPES: list[tuple[int, ...]] = [
    (3,),
    (2, 2),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5),
    (8, 7, 6, 5, 6, 7, 8),
    (10, 10, 10),
]

SMALL_TEST_SCALARS: tuple[float, ...] = (1, -1, math.pi)
TEST_SCALARS: tuple[float, ...] = (1, 2, 0.5, -1, 0, math.pi, -math.e, 1e30, -1e30)

type EpsilonComparable = NDArray | TTD[np.float64] | Scalar


def assert_default_epsilon(
    a: EpsilonComparable, b: EpsilonComparable, scale: EpsilonComparable = 1.0
) -> None:
    """
    Compare two tensors for equality within the default epsilon.

    Implicitly expands TTDs to ndarrays for comparison.

    Parameters
    ----------
    a
        The first tensor to compare.
    b
        The second tensor to compare.
    scale
        The original scale of the tensors, used to normalize the comparison.

    """
    scale = np.linalg.norm(np.asarray(scale))
    if scale <= np.finfo(np.float64).eps:
        scale = 1.0

    np.testing.assert_allclose(
        np.asarray(a) / scale,
        np.asarray(b) / scale,
        atol=DEFAULT_EPSILON,
    )


def tensor_interior(tensor: NDArray, order: int = 1) -> NDArray:
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


def tensor_boundary(tensor: NDArray, order: int = 1) -> NDArray:
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
