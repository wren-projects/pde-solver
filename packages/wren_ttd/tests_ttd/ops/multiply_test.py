# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np

from ..common import EpsilonComparable
from ..templates import BinaryOperatorTests, ScalarOperatorTests

if TYPE_CHECKING:
    from _typeshed import SupportsMul


class TestMultiplication(BinaryOperatorTests):
    """Tests for TTD elementwise multiplication."""

    @override
    def operator(self, a: EpsilonComparable, b: EpsilonComparable) -> EpsilonComparable:
        return a * b

    @override
    def operator_numpy(
        self, a: EpsilonComparable, b: EpsilonComparable
    ) -> EpsilonComparable:
        return np.multiply(a, b)

    @override
    def operator_inplace(self, a: SupportsMul[Any, Any], b: EpsilonComparable) -> None:
        a *= b


class TestScalarMultiplication(ScalarOperatorTests):
    """Tests for TTD-scalar multiplication."""

    @override
    def operator(self, a: EpsilonComparable, b: EpsilonComparable) -> EpsilonComparable:
        return a * b

    @override
    def operator_numpy(
        self, a: EpsilonComparable, b: EpsilonComparable
    ) -> EpsilonComparable:
        return np.multiply(a, b)

    @override
    def operator_inplace(self, a: SupportsMul[Any, Any], b: EpsilonComparable) -> None:
        a *= b
