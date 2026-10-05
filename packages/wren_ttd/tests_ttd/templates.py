from abc import ABC, abstractmethod
from copy import deepcopy

import pytest
from wren_common.tests import TEST_SCALARS, TestTensor, TestTensorPair

from .common import (
    TEST_PAIR_TTD,
    TEST_TTD,
    EpsilonComparable,
    TestTTD,
    TestTTDPair,
    assert_default_epsilon,
)


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
class BinaryOperatorTests(ABC):
    """Minimal interface for binary TTD-TTD operator-style op tests."""

    @abstractmethod
    def operator(self, a: EpsilonComparable, b: EpsilonComparable) -> EpsilonComparable:
        """Perform the op using the operator overloading on TTD."""
        ...

    @abstractmethod
    def operator_numpy(
        self, a: EpsilonComparable, b: EpsilonComparable
    ) -> EpsilonComparable:
        """Perform the op using the NumPy function."""
        ...

    @abstractmethod
    def operator_inplace(self, a: EpsilonComparable, b: EpsilonComparable) -> None:
        """Perform the op using the in-place operator."""
        ...

    def test_operator(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the Python operator as ``a op b``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(self.operator(ttd_a, ttd_b), self.operator(a, b))

    def test_numpy(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the NumPy function as ``f(a, b)``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(
            self.operator_numpy(ttd_a, ttd_b), self.operator_numpy(a, b)
        )

    def test_operator_matches_numpy(
        self,
        tensors: TestTensorPair,  # noqa: ARG002
        ttds: TestTTDPair,
    ) -> None:
        """Test the op via the Python operator as ``a op b``."""
        ttd_a, ttd_b = ttds
        assert_default_epsilon(
            self.operator(ttd_a, ttd_b), self.operator_numpy(ttd_a, ttd_b)
        )

    def test_inplace(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the in-place operator."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        ttd_copy = ttd_a.copy()
        self.operator_inplace(ttd_copy, ttd_b)
        assert_default_epsilon(ttd_copy, self.operator(a, b))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("scalar", TEST_SCALARS)
class ScalarOperatorTests(ABC):
    """Minimal interface for binary TTD-scalar operator-style op tests."""

    @abstractmethod
    def operator(self, a: EpsilonComparable, b: EpsilonComparable) -> EpsilonComparable:
        """Perform the op using the operator overloading on TTD."""
        ...

    @abstractmethod
    def operator_numpy(
        self, a: EpsilonComparable, b: EpsilonComparable
    ) -> EpsilonComparable:
        """Perform the op using the NumPy function."""
        ...

    @abstractmethod
    def operator_inplace(self, a: EpsilonComparable, b: EpsilonComparable) -> None:
        """Perform the op using the in-place operator."""
        ...

    def test_operator_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: float
    ) -> None:
        """Test the op via the Python operator as ``ttd op scalar``."""
        assert_default_epsilon(
            self.operator(ttd, scalar), self.operator(tensor, scalar)
        )

    def test_operator_left(
        self, tensor: TestTensor, ttd: TestTTD, scalar: float
    ) -> None:
        """Test the op via the Python operator as ``scalar op ttd``."""
        assert_default_epsilon(
            self.operator(scalar, ttd), self.operator(scalar, tensor)
        )

    def test_numpy_right(self, tensor: TestTensor, ttd: TestTTD, scalar: float) -> None:
        """Test the op via the NumPy function as ``f(ttd, scalar)``."""
        assert_default_epsilon(
            self.operator_numpy(ttd, scalar),
            self.operator_numpy(tensor, scalar),
        )

    def test_numpy_left(self, tensor: TestTensor, ttd: TestTTD, scalar: float) -> None:
        """Test the op via the NumPy function as ``f(scalar, ttd)``."""
        assert_default_epsilon(
            self.operator_numpy(scalar, ttd),
            self.operator_numpy(scalar, tensor),
        )

    def test_operator_matches_numpy_left(
        self,
        tensor: TestTensor,  # noqa: ARG002
        ttd: TestTTD,
        scalar: float,
    ) -> None:
        """Test the op via the Python operator as ``ttd op scalar``."""
        assert_default_epsilon(
            self.operator(ttd, scalar), self.operator_numpy(ttd, scalar)
        )

    def test_operator_matches_numpy_right(
        self,
        tensor: TestTensor,  # noqa: ARG002
        ttd: TestTTD,
        scalar: float,
    ) -> None:
        """Test the op via the Python operator as ``scalar op ttd``."""
        assert_default_epsilon(
            self.operator(scalar, ttd), self.operator_numpy(scalar, ttd)
        )

    def test_inplace(self, tensor: TestTensor, ttd: TestTTD, scalar: float) -> None:
        """Test the op via the in-place operator."""
        ttd_copy = ttd.copy()
        self.operator_inplace(ttd_copy, scalar)
        assert_default_epsilon(ttd_copy, self.operator(tensor, scalar))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class UnaryOperatorTests(ABC):
    """Minimal interface for unary operator-style op tests."""

    @abstractmethod
    def operator(self, a: EpsilonComparable) -> EpsilonComparable:
        """Perform the op using the operator overloading on TTD."""
        ...

    @abstractmethod
    def operator_numpy(self, a: EpsilonComparable) -> EpsilonComparable:
        """Perform the op using the NumPy function."""
        ...

    def test_operator(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test the op via the Python operator."""
        assert_default_epsilon(self.operator(ttd), self.operator(tensor))

    def test_numpy(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test the op via the NumPy function."""
        assert_default_epsilon(self.operator_numpy(ttd), self.operator_numpy(tensor))

    def test_operator_matches_numpy(self, tensor: TestTensor, ttd: TestTTD) -> None:  # noqa: ARG002
        """Test the op via the Python operator."""
        assert_default_epsilon(self.operator(ttd), self.operator_numpy(ttd))
