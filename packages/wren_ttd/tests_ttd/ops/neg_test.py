# pyright: reportAny=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import override

import numpy as np

from ..common import EpsilonComparable
from ..templates import UnaryOperatorTests


class TestNegation(UnaryOperatorTests):
    """Tests for TTD negation."""

    @override
    def operator(self, a: EpsilonComparable) -> EpsilonComparable:
        return -a

    @override
    def operator_numpy(self, a: EpsilonComparable) -> EpsilonComparable:
        return np.negative(a)
