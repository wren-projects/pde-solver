# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np

from ..common import EpsilonComparable
from ..templates import BinaryOperatorTests, ScalarOperatorTests

if TYPE_CHECKING:
    from _typeshed import SupportsSub


class TestSub(BinaryOperatorTests):
    """Tests for TTD subtraction."""

    @override
    def operator(self, a: EpsilonComparable, b: EpsilonComparable) -> EpsilonComparable:
        return a - b

    @override
    def operator_numpy(
        self, a: EpsilonComparable, b: EpsilonComparable
    ) -> EpsilonComparable:
        return np.subtract(a, b)

    @override
    def operator_inplace(self, a: SupportsSub[Any, Any], b: EpsilonComparable) -> None:
        a -= b


class TestScalarSubtraction(ScalarOperatorTests):
    """Tests for TTD-scalar subtraction."""

    @override
    def operator(self, a: EpsilonComparable, b: EpsilonComparable) -> EpsilonComparable:
        return a - b

    @override
    def operator_numpy(
        self, a: EpsilonComparable, b: EpsilonComparable
    ) -> EpsilonComparable:
        return np.subtract(a, b)

    @override
    def operator_inplace(self, a: SupportsSub[Any, Any], b: EpsilonComparable) -> None:
        a -= b
