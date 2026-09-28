from abc import ABC, abstractmethod

import numpy as np
from wren_common.tests import (
    TEST_COMPLEX_PAIR_TENSORS,
    TEST_COMPLEX_SCALARS,
    TEST_COMPLEX_TENSORS,
    TEST_PAIR_TENSORS,
    TEST_SCALARS,
    TEST_TENSORS,
    TestTensor,
    TestTensorPair,
)
from wren_common.types import NDArray, Scalar
from wren_ttd import DEFAULT_EPSILON, TTD

type TestTTD = TTD[np.float64]
type TestTTDPair = tuple[TTD[np.float64], TTD[np.float64]]

TEST_ALL_SCALARS: tuple[float | complex, ...] = (*TEST_SCALARS, *TEST_COMPLEX_SCALARS)


class BinaryOperatorTests(ABC):
    """Minimal interface for binary TTD-TTD operator-style op tests."""

    @abstractmethod
    def test_operator_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the Python operator as ``a op b``."""
        ...

    @abstractmethod
    def test_operator_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the Python operator as ``b op a``."""
        ...

    @abstractmethod
    def test_numpy_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the NumPy function as ``f(a, b)``."""
        ...

    @abstractmethod
    def test_numpy_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the NumPy function as ``f(b, a)``."""
        ...

    @abstractmethod
    def test_inplace(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the in-place operator."""
        ...


class ScalarOperatorTests(ABC):
    """Minimal interface for binary TTD-scalar operator-style op tests."""

    @abstractmethod
    def test_operator_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test the op via the Python operator as ``ttd op scalar``."""
        ...

    @abstractmethod
    def test_operator_left(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test the op via the Python operator as ``scalar op ttd``."""
        ...

    @abstractmethod
    def test_numpy_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test the op via the NumPy function as ``f(ttd, scalar)``."""
        ...

    @abstractmethod
    def test_numpy_left(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test the op via the NumPy function as ``f(scalar, ttd)``."""
        ...

    @abstractmethod
    def test_inplace(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test the op via the in-place operator."""
        ...


class UnaryOperatorTests(ABC):
    """Minimal interface for unary operator-style op tests."""

    @abstractmethod
    def test_operator(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test the op via the Python operator."""
        ...

    @abstractmethod
    def test_numpy(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test the op via the NumPy function."""
        ...


TEST_TTD: list[tuple[TestTensor, TestTTD]] = [
    (tensor, TTD.from_ndarray(tensor)) for tensor in TEST_TENSORS
] + [(tensor, TTD.from_ndarray(tensor)) for tensor in TEST_COMPLEX_TENSORS]

TEST_PAIR_TTD: list[tuple[TestTensorPair, TestTTDPair]] = [
    ((a, b), (TTD.from_ndarray(a), TTD.from_ndarray(b))) for a, b in TEST_PAIR_TENSORS
] + [
    ((a, b), (TTD.from_ndarray(a), TTD.from_ndarray(b)))
    for a, b in TEST_COMPLEX_PAIR_TENSORS
]


type EpsilonComparable = NDArray | TTD | Scalar


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
