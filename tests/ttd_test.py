import numpy as np
import pytest

from numpy_ttd import TTD
from tests.common import (
    TEST_PAIR_TTD,
    TEST_SCALARS,
    TEST_TTD,
    TestTensor,
    TestTensorPair,
    TestTTD,
    TestTTDPair,
    assert_default_epsilon,
)


@pytest.mark.parametrize(("tensor", "ttd"), TEST_TTD)
def test_roundtrip_compression(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test the round-trip compression/decompression of a tensor."""
    assert_default_epsilon(ttd, tensor)


@pytest.mark.parametrize(("tensor", "ttd"), TEST_TTD)
def test_inner_product(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test inner norm."""
    assert_default_epsilon(np.vdot(ttd, ttd), np.vdot(tensor, tensor))


@pytest.mark.parametrize(("tensor", "ttd"), TEST_TTD)
def test_frobenius_norm(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test Frobenius norm."""
    assert_default_epsilon(np.linalg.norm(ttd), np.linalg.norm(tensor))


@pytest.mark.parametrize(("tensor", "ttd"), TEST_TTD)
def test_rounding(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test get value."""
    assert_default_epsilon(ttd.rounded(), tensor)


@pytest.mark.parametrize(("tensor", "ttd"), TEST_TTD)
def test_get_element(ttd: TestTTD, tensor: TestTensor) -> None:
    """Test get value."""
    for index in np.ndindex(tensor.shape):
        assert_default_epsilon(ttd[index], tensor[index])


@pytest.mark.parametrize(("tensors", "ttds"), TEST_PAIR_TTD)
def test_add(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test that TTD addition works."""
    a, b = tensors
    ttd_a, ttd_b = ttds

    assert a.shape == b.shape
    tensor_sum = a + b

    assert_default_epsilon(ttd_a + ttd_b, tensor_sum)
    assert_default_epsilon(ttd_b + ttd_a, tensor_sum)
    assert_default_epsilon(np.add(ttd_a, ttd_b), tensor_sum)
    assert_default_epsilon(np.add(ttd_b, ttd_a), tensor_sum)

    ttd_copy = ttd_a[...]
    ttd_copy += ttd_b
    assert_default_epsilon(ttd_copy, tensor_sum)


@pytest.mark.parametrize(("tensors", "ttds"), TEST_PAIR_TTD)
def test_sub(tensors: TestTensorPair, ttds: TestTTDPair) -> None:
    """Test that TTD addition works."""
    a, b = tensors
    ttd_a, ttd_b = ttds

    assert a.shape == b.shape
    tensor_diff = a - b

    assert_default_epsilon(ttd_a - ttd_b, tensor_diff)
    assert_default_epsilon(ttd_b - ttd_a, -tensor_diff)
    assert_default_epsilon(np.subtract(ttd_a, ttd_b), tensor_diff)
    assert_default_epsilon(np.subtract(ttd_b, ttd_a), -tensor_diff)

    ttd_copy = ttd_a[...]
    ttd_copy -= ttd_b
    assert_default_epsilon(ttd_copy, tensor_diff)


@pytest.mark.parametrize(("tensor", "ttd"), TEST_TTD)
@pytest.mark.parametrize(("scalar"), TEST_SCALARS)
def test_scalar_multiplication(tensor: TestTensor, ttd: TestTTD, scalar: float) -> None:
    """Test that TTD scalar multiplication works."""
    scaled_tensor = tensor * scalar
    assert_default_epsilon(ttd * scalar, scaled_tensor)
    assert_default_epsilon(scalar * ttd, scaled_tensor)
    assert_default_epsilon(np.multiply(ttd, scalar), scaled_tensor)
    assert_default_epsilon(np.multiply(scalar, ttd), scaled_tensor)

    ttd_copy = ttd[...]
    ttd_copy *= scalar
    assert_default_epsilon(ttd_copy, scaled_tensor)


@pytest.mark.parametrize(("tensor", "ttd"), TEST_TTD)
def test_negation(tensor: TestTensor, ttd: TestTTD) -> None:
    """Test that TTD negation works."""
    assert_default_epsilon(-ttd, -tensor)
    assert_default_epsilon(np.negative(ttd), np.negative(tensor))
