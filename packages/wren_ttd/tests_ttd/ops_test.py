from copy import deepcopy
from typing import cast

import numpy as np
import pytest
from wren_common.tests import (
    TEST_SHAPES,
    TestTensor,
    TestTensorPair,
)
from wren_common.types import Scalar
from wren_ttd import TTD

from .common import (
    TEST_ALL_SCALARS,
    TEST_PAIR_TTD,
    TEST_TTD,
    BinaryOperatorTests,
    ScalarOperatorTests,
    TestTTD,
    TestTTDPair,
    UnaryOperatorTests,
    assert_default_epsilon,
)


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
def test_inner_product(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test inner product."""
    ttd_a, ttd_b = ttds
    tensor_a, tensor_b = tensors

    assert_default_epsilon(np.vdot(ttd_a, ttd_b), np.vdot(tensor_a, tensor_b))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
def test_frobenius_norm(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test Frobenius norm."""
    assert_default_epsilon(np.linalg.norm(ttd), np.linalg.norm(tensor))


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_zeros(shape: tuple[int, ...]) -> None:
    """Test zeros."""
    ttd = TTD.zeros(shape, dtype=np.dtype(np.float64))
    tensor = np.zeros(shape, dtype=np.dtype(np.float64))
    assert_default_epsilon(ttd, tensor)


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_ones(shape: tuple[int, ...]) -> None:
    """Test ones."""
    ttd = TTD.ones(shape, dtype=np.dtype(np.float64))
    tensor = np.ones(shape, dtype=np.dtype(np.float64))
    assert_default_epsilon(ttd, tensor)


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
@pytest.mark.parametrize("fill_value", deepcopy(TEST_ALL_SCALARS))
def test_full(shape: tuple[int, ...], fill_value: Scalar) -> None:
    """Test full."""
    ttd = TTD.full(shape, fill_value)
    tensor = np.full(shape, fill_value)
    assert_default_epsilon(ttd, tensor)


@pytest.mark.parametrize("shape", deepcopy(TEST_SHAPES))
def test_random(shape: tuple[int, ...]) -> None:
    """Test randomly generated TTD."""
    ttd = TTD.random(shape, dtype=np.dtype(np.float64))
    assert ttd.shape == shape
    np.testing.assert_allclose(ttd.sum() / ttd.size, 0.5, atol=0.5, rtol=0.5)


def test_ranks() -> None:
    """Test ranks."""
    ttd = TTD([np.zeros((1, 2, 2)), np.zeros((2, 3, 2)), np.zeros((2, 2, 1))])
    assert ttd.ranks == (2, 2)


def test_compressed_size() -> None:
    """Test ranks."""
    ttd = TTD([np.zeros((1, 2, 2)), np.zeros((2, 3, 2)), np.zeros((2, 2, 1))])
    assert ttd.compressed_size == 2 * 2 + 2 * 3 * 2 + 2 * 2


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestRounding:
    """Tests for TTD rounding."""

    def test_plain(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test rounding a plain TTD."""
        assert_default_epsilon(ttd.rounded(), tensor)

    def test_with_inflated_ranks(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test rounding a TTD with inflated ranks."""
        added = ttd + ttd
        rounded = added.rounded()
        assert_default_epsilon(rounded, 2 * tensor)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestSum:
    """Tests for TTD summation."""

    def test_full(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test summation over all axes."""
        assert_default_epsilon(np.sum(ttd), np.sum(tensor))

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_single_axis(self, tensor: TestTensor, ttd: TestTTD, axis: int) -> None:
        """Test summation over a single axis."""
        assert_default_epsilon(ttd.sum(axis), cast(TestTensor, tensor.sum(axis)))

    def test_axes_tuple(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test summation over a tuple of axes."""
        assert_default_epsilon(ttd.sum((0, 1)), cast(TestTensor, tensor.sum((0, 1))))


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
class TestAdd(BinaryOperatorTests):
    """Tests for TTD addition."""

    def test_operator_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test addition via the ``+`` operator."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(ttd_a + ttd_b, a + b)

    def test_operator_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test addition via the ``+`` operator with swapped operands."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(ttd_b + ttd_a, a + b)

    def test_numpy_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test addition via ``np.add``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(np.add(ttd_a, ttd_b), a + b)

    def test_numpy_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test addition via ``np.add`` with swapped operands."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(np.add(ttd_b, ttd_a), a + b)

    def test_inplace(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test in-place addition via ``+=``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        ttd_copy = ttd_a.copy()
        ttd_copy += ttd_b
        assert_default_epsilon(ttd_copy, a + b)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("scalar", TEST_ALL_SCALARS)
class TestScalarAddition(ScalarOperatorTests):
    """Tests for TTD-scalar addition."""

    def test_operator_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test scalar addition via ``ttd + scalar``."""
        assert_default_epsilon(ttd + scalar, tensor + scalar)

    def test_operator_left(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test scalar addition via ``scalar + ttd``."""
        assert_default_epsilon(scalar + ttd, tensor + scalar)

    def test_numpy_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test scalar addition via ``np.add(ttd, scalar)``."""
        assert_default_epsilon(np.add(ttd, scalar), tensor + scalar)

    def test_numpy_left(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test scalar addition via ``np.add(scalar, ttd)``."""
        assert_default_epsilon(np.add(scalar, ttd), tensor + scalar)

    def test_inplace(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test in-place scalar addition via ``+=``."""
        summed_tensor = tensor + scalar
        scale = np.linalg.norm(summed_tensor)
        ttd_copy = ttd.copy()
        if np.can_cast(summed_tensor.dtype, ttd_copy.dtype, casting="same_kind"):
            ttd_copy += scalar
            assert_default_epsilon(ttd_copy, summed_tensor, scale)
        else:
            # Mirror NumPy: an in-place op cannot upcast the output dtype.
            with pytest.raises(TypeError, match="same_kind"):
                ttd_copy += scalar


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
class TestSub(BinaryOperatorTests):
    """Tests for TTD subtraction."""

    def test_operator_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test subtraction via the ``-`` operator."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(ttd_a - ttd_b, a - b)

    def test_operator_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test subtraction via the ``-`` operator with swapped operands."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(ttd_b - ttd_a, b - a)

    def test_numpy_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test subtraction via ``np.subtract``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(np.subtract(ttd_a, ttd_b), a - b)

    def test_numpy_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test subtraction via ``np.subtract`` with swapped operands."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(np.subtract(ttd_b, ttd_a), b - a)

    def test_inplace(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test in-place subtraction via ``-=``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        ttd_copy = ttd_a.copy()
        ttd_copy -= ttd_b
        assert_default_epsilon(ttd_copy, a - b)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("scalar", TEST_ALL_SCALARS)
class TestScalarSubtraction(ScalarOperatorTests):
    """Tests for TTD-scalar subtraction."""

    def test_operator_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test scalar subtraction via ``ttd - scalar``."""
        assert_default_epsilon(ttd - scalar, tensor - scalar)

    def test_operator_left(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test scalar subtraction via ``scalar - ttd``."""
        assert_default_epsilon(scalar - ttd, scalar - tensor)

    def test_numpy_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test scalar subtraction via ``np.subtract(ttd, scalar)``."""
        assert_default_epsilon(np.subtract(ttd, scalar), tensor - scalar)

    def test_numpy_left(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test scalar subtraction via ``np.subtract(scalar, ttd)``."""
        assert_default_epsilon(np.subtract(scalar, ttd), scalar - tensor)

    def test_inplace(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test in-place scalar subtraction via ``-=``."""
        subtracted_tensor = tensor - scalar
        scale = np.linalg.norm(subtracted_tensor)
        ttd_copy = ttd.copy()
        if np.can_cast(subtracted_tensor.dtype, ttd_copy.dtype, casting="same_kind"):
            ttd_copy -= scalar
            assert_default_epsilon(ttd_copy, subtracted_tensor, scale)
        else:
            # Mirror NumPy: an in-place op cannot upcast the output dtype.
            with pytest.raises(TypeError, match="same_kind"):
                ttd_copy -= scalar


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
class TestMultiplication(BinaryOperatorTests):
    """Tests for TTD elementwise multiplication."""

    def test_operator_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test multiplication via the ``*`` operator."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(ttd_a * ttd_b, a * b)

    def test_operator_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test multiplication via the ``*`` operator with swapped operands."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(ttd_b * ttd_a, a * b)

    def test_numpy_ab(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test multiplication via ``np.multiply``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(np.multiply(ttd_a, ttd_b), a * b)

    def test_numpy_ba(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test multiplication via ``np.multiply`` with swapped operands."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(np.multiply(ttd_b, ttd_a), a * b)

    def test_inplace(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test in-place multiplication via ``*=``."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        ttd_copy = ttd_a.copy()
        ttd_copy *= ttd_b
        assert_default_epsilon(ttd_copy, a * b)


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("scalar", deepcopy(TEST_ALL_SCALARS))
class TestScalarMultiplication(ScalarOperatorTests):
    """Tests for TTD-scalar multiplication."""

    def test_operator_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test scalar multiplication via ``ttd * scalar``."""
        scaled_tensor = tensor * scalar
        scale = np.linalg.norm(scaled_tensor)
        assert_default_epsilon(ttd * scalar, scaled_tensor, scale)

    def test_operator_left(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test scalar multiplication via ``scalar * ttd``."""
        scaled_tensor = tensor * scalar
        scale = np.linalg.norm(scaled_tensor)
        assert_default_epsilon(scalar * ttd, scaled_tensor, scale)

    def test_numpy_right(
        self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar
    ) -> None:
        """Test scalar multiplication via ``np.multiply(ttd, scalar)``."""
        scaled_tensor = tensor * scalar
        scale = np.linalg.norm(scaled_tensor)
        assert_default_epsilon(np.multiply(ttd, scalar), scaled_tensor, scale)

    def test_numpy_left(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test scalar multiplication via ``np.multiply(scalar, ttd)``."""
        scaled_tensor = tensor * scalar
        scale = np.linalg.norm(scaled_tensor)
        assert_default_epsilon(np.multiply(scalar, ttd), scaled_tensor, scale)

    def test_inplace(self, tensor: TestTensor, ttd: TestTTD, scalar: Scalar) -> None:
        """Test in-place scalar multiplication via ``*=``."""
        scaled_tensor = tensor * scalar
        scale = np.linalg.norm(scaled_tensor)
        ttd_copy = ttd.copy()
        if np.can_cast(scaled_tensor.dtype, ttd_copy.dtype, casting="same_kind"):
            ttd_copy *= scalar
            assert_default_epsilon(ttd_copy, scaled_tensor, scale)
        else:
            # Mirror NumPy: an in-place op cannot upcast the output dtype.
            with pytest.raises(TypeError, match="same_kind"):
                ttd_copy *= scalar


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestNegation(UnaryOperatorTests):
    """Tests for TTD negation."""

    def test_operator(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test negation via the ``-`` operator."""
        assert_default_epsilon(-ttd, -tensor)

    def test_numpy(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test negation via ``np.negative``."""
        assert_default_epsilon(np.negative(ttd), np.negative(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestTranspose:
    """Tests for TTD transpose."""

    def test_default_method(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose via the ``transpose`` method."""
        assert_default_epsilon(ttd.transpose(), tensor.transpose())

    def test_property_t(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose via the ``T`` property."""
        assert_default_epsilon(ttd.T, tensor.T)

    def test_swap_first_two(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose swapping the first two axes."""
        axes = (1, 0, *range(2, tensor.ndim))
        assert_default_epsilon(
            np.transpose(ttd, axes),
            np.transpose(tensor, axes),
        )

    def test_rotate(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose rotating axes to the left."""
        axes = (*range(1, tensor.ndim), 0)
        assert_default_epsilon(
            np.transpose(ttd, axes),
            np.transpose(tensor, axes),
        )

    def test_last_to_front(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose moving the last axis to the front."""
        axes = (-1, *range(tensor.ndim - 1))
        assert_default_epsilon(
            np.transpose(ttd, axes),
            np.transpose(tensor, axes),
        )

    def test_reverse(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test transpose reversing all axes."""
        axes = tuple(reversed(range(tensor.ndim)))
        assert_default_epsilon(
            np.transpose(ttd, axes),
            np.transpose(tensor, axes),
        )


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize(
    "axes",
    [(0, 1), (0, -1), (-1, -2), (0, 0)],
)
class TestSwapaxes:
    """Tests for TTD swapaxes."""

    def test_numpy(
        self, tensor: TestTensor, ttd: TestTTD, axes: tuple[int, int]
    ) -> None:
        """Test swapaxes via ``np.swapaxes``."""
        axis1, axis2 = axes
        assert_default_epsilon(
            np.swapaxes(ttd, axis1, axis2), np.swapaxes(tensor, axis1, axis2)
        )

    def test_method(
        self, tensor: TestTensor, ttd: TestTTD, axes: tuple[int, int]
    ) -> None:
        """Test swapaxes via the ``swapaxes`` method."""
        axis1, axis2 = axes
        assert_default_epsilon(
            ttd.swapaxes(axis1, axis2), tensor.swapaxes(axis1, axis2)
        )


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
class TestTensordot:
    """Tests for TTD tensordot."""

    def test_axes_0(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test tensordot with ``axes=0`` (outer product)."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(
            np.tensordot(ttd_a, ttd_b, axes=0),
            np.tensordot(a, b, axes=0),
        )

    def test_axes_1_transposed(
        self, tensors: TestTensorPair, ttds: TestTTDPair
    ) -> None:
        """Test tensordot with ``axes=1`` against a transposed operand."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(
            np.tensordot(ttd_a, ttd_b.T, axes=1),
            np.tensordot(a, b.T, axes=1),
        )

    def test_axes_1_permuted(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test tensordot with ``axes=1`` against a permuted operand."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        axes = (1, 0, *range(2, ttd_a.ndim))
        assert_default_epsilon(
            np.tensordot(ttd_a, np.transpose(ttd_b.T, axes=axes)),
            np.tensordot(a, np.transpose(b.T, axes=axes)),
        )

    def test_axes_00(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test tensordot contracting the first axes."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(
            np.tensordot(ttd_a, ttd_b, axes=(0, 0)),
            np.tensordot(a, b, axes=(0, 0)),
        )

    def test_axes_minus1_minus1(
        self, tensors: TestTensorPair, ttds: TestTTDPair
    ) -> None:
        """Test tensordot contracting the last axes."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(
            np.tensordot(ttd_a, ttd_b, axes=(-1, -1)),
            np.tensordot(a, b, axes=(-1, -1)),
        )

    def test_axes_11(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test tensordot contracting the second axes."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(
            np.tensordot(ttd_a, ttd_b, axes=(1, 1)),
            np.tensordot(a, b, axes=(1, 1)),
        )

    def test_axes_all_but_last(
        self, tensors: TestTensorPair, ttds: TestTTDPair
    ) -> None:
        """Test tensordot contracting all but the last axes."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        axes_single = tuple(range(a.ndim - 1))
        axes_pair = (axes_single, axes_single)
        assert_default_epsilon(
            np.tensordot(ttd_a, ttd_b, axes=axes_pair),
            np.tensordot(a, b, axes=axes_pair),
        )

    def test_axes_all(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test tensordot contracting all axes."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        axes_single = tuple(range(a.ndim))
        axes_pair = (axes_single, axes_single)
        assert_default_epsilon(
            np.tensordot(ttd_a, ttd_b, axes=axes_pair),
            np.tensordot(a, b, axes=axes_pair),
        )


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestStackSingle:
    """Tests for stacking a single TTD."""

    def test_stack(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test stacking a single tensor."""
        assert_default_epsilon(
            np.stack(ttd),
            np.stack(tensor),
        )


@pytest.mark.parametrize(("tensors", "ttds"), deepcopy(TEST_PAIR_TTD))
class TestStack:
    """Tests for stacking TTD pairs."""

    @pytest.mark.parametrize("axis", [0, -1, 1])
    @pytest.mark.parametrize("repeat", [1, 2])
    def test_stack(
        self, tensors: TestTensorPair, ttds: TestTTDPair, axis: int, repeat: int
    ) -> None:
        """Test stack for the selected axis, with optional repetition."""
        assert_default_epsilon(
            np.stack(ttds * repeat, axis=axis),
            np.stack(tensors * repeat, axis=axis),
        )


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
class TestGradient:
    """Tests for TTD gradient."""

    def test_default(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient with default arguments."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(np.gradient(ttd), np.gradient(tensor), scale)

    def test_axis_1(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient along a single axis."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(
            np.gradient(ttd, axis=1), np.gradient(tensor, axis=1), scale
        )

    def test_axis_range(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient along a range of axes."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(
            np.gradient(ttd, axis=range(1, ttd.ndim)),
            np.gradient(tensor, axis=range(1, tensor.ndim)),
            scale,
        )

    def test_axis_negative_range(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient along a range of negative axes."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(
            np.gradient(ttd, axis=range(-ttd.ndim + 1, 0)),
            np.gradient(tensor, axis=range(-tensor.ndim + 1, 0)),
            scale,
        )

    def test_uniform_steps(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient with uniform scalar steps."""
        scale = np.linalg.norm(tensor)
        steps = tuple((n + 1) / 10 for n in range(ttd.ndim))
        assert_default_epsilon(
            np.gradient(ttd, *steps), np.gradient(tensor, *steps), scale
        )

    def test_array_steps(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient with array steps."""
        scale = np.linalg.norm(tensor)
        steps = tuple(tuple((x + 1) / 10 for x in range(n)) for n in ttd.shape)
        assert_default_epsilon(
            np.gradient(ttd, *steps), np.gradient(tensor, *steps), scale
        )

    def test_edge_order_2(self, tensor: TestTensor, ttd: TestTTD) -> None:
        """Test gradient with second-order edge differences."""
        scale = np.linalg.norm(tensor)
        assert_default_epsilon(
            np.gradient(ttd, edge_order=2), np.gradient(tensor, edge_order=2), scale
        )


@pytest.mark.parametrize(("tensor", "ttd"), deepcopy(TEST_TTD))
@pytest.mark.parametrize("value", TEST_ALL_SCALARS)
@pytest.mark.parametrize("width", [0, 1, 2])
class TestPad:
    """Tests for TTD padding."""

    def test_pad_single_constant(
        self, tensor: TestTensor, ttd: TestTTD, value: Scalar, width: int
    ) -> None:
        """Test pad with the selected constant value in single width."""
        expected = np.pad(tensor, width, mode="constant", constant_values=value)
        result = np.pad(ttd, width, mode="constant", constant_values=value)
        assert_default_epsilon(result, expected, scale=value)
