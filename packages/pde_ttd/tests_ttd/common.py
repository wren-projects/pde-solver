import numpy as np
from pde_common.tests import (
    TEST_PAIR_TENSORS,
    TEST_TENSORS,
    TestTensor,
    TestTensorPair,
)
from pde_common.types import NDArray
from pde_ttd import DEFAULT_EPSILON, TTD

type TestTTD = TTD[np.float64]
type TestTTDPair = tuple[TTD[np.float64], TTD[np.float64]]

TEST_TTD: list[tuple[TestTensor, TestTTD]] = [
    (tensor, TTD.from_ndarray(tensor)) for tensor in TEST_TENSORS
]

TEST_PAIR_TTD: list[tuple[TestTensorPair, TestTTDPair]] = [
    ((a, b), (TTD.from_ndarray(a), TTD.from_ndarray(b))) for a, b in TEST_PAIR_TENSORS
]


type EpsilonComparable = NDArray[np.floating] | TTD[np.floating] | np.floating | float


def assert_default_epsilon(
    a: EpsilonComparable,
    b: EpsilonComparable,
    scale: EpsilonComparable = 1.0,
    epsilon: float = DEFAULT_EPSILON,
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
    epsilon
        The epsilon to use for the comparison.

    """
    scale = np.linalg.norm(np.asarray(scale))
    if scale <= np.finfo(np.float64).eps:
        scale = 1.0

    np.testing.assert_allclose(
        np.asarray(a) / scale,
        np.asarray(b) / scale,
        atol=DEFAULT_EPSILON,
        rtol=epsilon,
    )
