import operator as operator_module
from abc import ABC
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pytest
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

type EpsilonComparable = NDArray | TTD | Scalar

TEST_ALL_SCALARS: tuple[float | complex, ...] = (*TEST_SCALARS, *TEST_COMPLEX_SCALARS)


def _derive_inplace_op(op: Callable[..., Any]) -> Callable[..., Any]:
    """Derive ``operator.i*`` from ``operator.*`` (e.g. ``add`` -> ``iadd``)."""
    name = getattr(op, "__name__", None)
    if name is None:
        raise TypeError(f"Cannot derive inplace op from {op!r}: no __name__.")
    try:
        return getattr(operator_module, "i" + name)
    except AttributeError as e:
        raise TypeError(
            f"Cannot derive inplace op from {name!r}: "
            f"operator.i{name} does not exist. Set inplace_op explicitly."
        ) from e


def _ensure_binary_attrs(cls: type) -> None:
    """Validate metadata on BinaryOperatorTests subclasses, deriving inplace."""
    if "op" not in cls.__dict__ and not hasattr(cls, "op"):
        raise TypeError(f"{cls.__name__} must define class attribute 'op'.")
    if "numpy_ufunc" not in cls.__dict__ and not hasattr(cls, "numpy_ufunc"):
        raise TypeError(f"{cls.__name__} must define class attribute 'numpy_ufunc'.")
    if "inplace_op" not in cls.__dict__:
        # Derive per-subclass so an overridden op never reuses a stale inplace.
        cls.inplace_op = _derive_inplace_op(cls.op)  # type: ignore[attr-defined]


def _ensure_scalar_attrs(cls: type) -> None:
    """Validate metadata on ScalarOperatorTests subclasses, deriving inplace."""
    _ensure_binary_attrs(cls)


def _ensure_unary_attrs(cls: type) -> None:
    """Validate metadata on UnaryOperatorTests subclasses."""
    if "op" not in cls.__dict__ and not hasattr(cls, "op"):
        raise TypeError(f"{cls.__name__} must define class attribute 'op'.")
    if "numpy_ufunc" not in cls.__dict__ and not hasattr(cls, "numpy_ufunc"):
        raise TypeError(f"{cls.__name__} must define class attribute 'numpy_ufunc'.")


class BinaryOperatorTests(ABC):
    """
    Template for binary TTD-TTD operator-style op tests.

    Subclasses only declare metadata; the five test methods are provided::

        class TestAdd(BinaryOperatorTests):
            op = operator.add
            numpy_ufunc = np.add
            # inplace_op defaults to operator.iadd, derived from op

    Set ``inplace_op`` explicitly only when it cannot be derived
    (i.e. ``operator.i<op.__name__>`` does not exist).
    """

    op: ClassVar[Callable[[Any, Any], EpsilonComparable]]
    numpy_ufunc: ClassVar[Callable[..., EpsilonComparable]]
    inplace_op: ClassVar[Callable[[Any, Any], EpsilonComparable]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate metadata and derive the inplace operator."""
        super().__init_subclass__(**kwargs)
        _ensure_binary_attrs(cls)

    def test_operator_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the Python operator as ``a op b``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        cls = type(self)
        assert_default_epsilon(cls.op(ttd_a, ttd_b), cls.op(a, b))

    def test_operator_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the Python operator as ``b op a``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        cls = type(self)
        assert_default_epsilon(cls.op(ttd_b, ttd_a), cls.op(b, a))

    def test_numpy_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the NumPy function as ``f(a, b)``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        cls = type(self)
        assert_default_epsilon(cls.numpy_ufunc(ttd_a, ttd_b), cls.op(a, b))

    def test_numpy_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the NumPy function as ``f(b, a)``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        cls = type(self)
        assert_default_epsilon(cls.numpy_ufunc(ttd_b, ttd_a), cls.op(b, a))

    def test_inplace(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test the op via the in-place operator."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        cls = type(self)
        ttd_copy = ttd_a.copy()
        ttd_copy = cls.inplace_op(ttd_copy, ttd_b)
        assert_default_epsilon(ttd_copy, cls.op(a, b))


class ScalarOperatorTests(ABC):
    """
    Template for binary TTD-scalar operator-style op tests.

    Subclasses only declare metadata; the five test methods are provided::

        class TestScalarAddition(ScalarOperatorTests):
            op = operator.add
            numpy_ufunc = np.add
    """

    op: ClassVar[Callable[[Any, Any], EpsilonComparable]]
    numpy_ufunc: ClassVar[Callable[..., EpsilonComparable]]
    inplace_op: ClassVar[Callable[[Any, Any], EpsilonComparable]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate metadata and derive the inplace operator."""
        super().__init_subclass__(**kwargs)
        _ensure_scalar_attrs(cls)

    def test_operator_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test the op via the Python operator as ``ttd op scalar``."""
        cls = type(self)
        expected = cls.op(tensor, scalar)
        assert_default_epsilon(
            cls.op(ttd, scalar), expected, scale=np.linalg.norm(expected)
        )

    def test_operator_left(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test the op via the Python operator as ``scalar op ttd``."""
        cls = type(self)
        expected = cls.op(scalar, tensor)
        assert_default_epsilon(
            cls.op(scalar, ttd), expected, scale=np.linalg.norm(expected)
        )

    def test_numpy_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test the op via the NumPy function as ``f(ttd, scalar)``."""
        cls = type(self)
        expected = cls.op(tensor, scalar)
        assert_default_epsilon(
            cls.numpy_ufunc(ttd, scalar), expected, scale=np.linalg.norm(expected)
        )

    def test_numpy_left(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test the op via the NumPy function as ``f(scalar, ttd)``."""
        cls = type(self)
        expected = cls.op(scalar, tensor)
        assert_default_epsilon(
            cls.numpy_ufunc(scalar, ttd), expected, scale=np.linalg.norm(expected)
        )

    def test_inplace(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test the op via the in-place operator."""
        cls = type(self)
        expected = cls.op(tensor, scalar)
        scale = np.linalg.norm(np.asarray(expected))
        ttd_copy = ttd.copy()
        if np.can_cast(np.asarray(expected).dtype, ttd_copy.dtype, casting="same_kind"):
            ttd_copy = cls.inplace_op(ttd_copy, scalar)
            assert_default_epsilon(ttd_copy, expected, scale)
        else:
            # Mirror NumPy: an in-place op cannot upcast the output dtype.
            with pytest.raises(TypeError, match="same_kind"):
                cls.inplace_op(ttd_copy, scalar)


class UnaryOperatorTests(ABC):
    """
    Template for unary operator-style op tests.

    Subclasses only declare metadata; the two test methods are provided::

        class TestNegation(UnaryOperatorTests):
            op = operator.neg
            numpy_ufunc = np.negative
    """

    op: ClassVar[Callable[[Any], EpsilonComparable]]
    numpy_ufunc: ClassVar[Callable[..., EpsilonComparable]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate subclass metadata."""
        super().__init_subclass__(**kwargs)
        _ensure_unary_attrs(cls)

    def test_operator(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test the op via the Python operator."""
        cls = type(self)
        assert_default_epsilon(cls.op(ttd), cls.op(tensor))

    def test_numpy(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test the op via the NumPy function."""
        cls = type(self)
        assert_default_epsilon(cls.numpy_ufunc(ttd), cls.numpy_ufunc(tensor))


TEST_TTD: list[tuple[TestTensor, TestTTD]] = [
    (tensor, TTD.from_ndarray(tensor)) for tensor in TEST_TENSORS
] + [(tensor, TTD.from_ndarray(tensor)) for tensor in TEST_COMPLEX_TENSORS]

TEST_PAIR_TTD: list[tuple[TestTensorPair, TestTTDPair]] = [
    ((a, b), (TTD.from_ndarray(a), TTD.from_ndarray(b))) for a, b in TEST_PAIR_TENSORS
] + [
    ((a, b), (TTD.from_ndarray(a), TTD.from_ndarray(b)))
    for a, b in TEST_COMPLEX_PAIR_TENSORS
]


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
