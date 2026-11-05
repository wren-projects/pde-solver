# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np

from ..common import EpsilonComparable
from ..templates import BinaryOperatorTests, ScalarOperatorTests

if TYPE_CHECKING:
    from _typeshed import SupportsAdd


class TestAdd(BinaryOperatorTests):
    """Tests for TTD addition."""

    @override
    def operator(self, a: EpsilonComparable, b: EpsilonComparable) -> EpsilonComparable:
        return a + b

    @override
    def operator_numpy(
        self, a: EpsilonComparable, b: EpsilonComparable
    ) -> EpsilonComparable:
        return np.add(a, b)

    @override
    def operator_inplace(self, a: SupportsAdd[Any, Any], b: EpsilonComparable) -> None:
        a += b


class TestScalarAddition(ScalarOperatorTests):
    """Tests for TTD-scalar addition."""

    @override
    def operator(self, a: EpsilonComparable, b: EpsilonComparable) -> EpsilonComparable:
        return a + b

    @override
    def operator_numpy(
        self, a: EpsilonComparable, b: EpsilonComparable
    ) -> EpsilonComparable:
        return np.add(a, b)

    @override
    def operator_inplace(self, a: SupportsAdd[Any, Any], b: EpsilonComparable) -> None:
        a += b
