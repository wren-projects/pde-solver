# pyright: reportAny=false, reportUnknownVariableType=false
from copy import deepcopy

import numpy as np
import pytest
from wren_common.tests import TestTensorPair

from ..common import (
    TEST_PAIR_TTD,
    TestTTDPair,
    assert_default_epsilon,
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

    def test_axes_1(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test tensordot with ``axes=1`` (dot product)."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        assert_default_epsilon(
            np.tensordot(ttd_a, ttd_b.T, axes=1),
            np.tensordot(a, b.T, axes=1),
        )

    def test_axes_default_2(self, tensors: TestTensorPair, ttds: TestTTDPair) -> None:
        """Test tensordot with default ``axes=2`` (double contraction)."""
        a, b = tensors
        ttd_a, ttd_b = ttds
        # the operand is permuted to match mode sizes
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
