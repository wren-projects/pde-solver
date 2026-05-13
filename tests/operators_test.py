from typing import cast
import numpy as np
from numpy.testing import assert_allclose
import pytest
import sympy as sp  # pyright: ignore[reportMissingTypeStubs]

from pde_solver.operators import gradient

a: sp.Symbol = sp.Symbol("a")
b: sp.Symbol = sp.Symbol("b")
c: sp.Symbol = sp.Symbol("c")
d: sp.Symbol = sp.Symbol("d")
e: sp.Symbol = sp.Symbol("e")
f: sp.Symbol = sp.Symbol("f")
variables = [a, b, c, d, e, f]


def _get_args(function: sp.Expr) -> list[sp.Symbol]:
    """
    Find which arguments the function takes.

    To work correctly with functions constant in some arguments, we assume that if a
    function takes f, it takes e, d,… too.
    """
    var_sequence = [v.name for v in variables]
    used_vars = function.free_symbols
    indices = [var_sequence.index(str(v)) for v in used_vars if str(v) in var_sequence]
    if not indices:
        return []
    return variables[: max(indices) + 1]


def _autocompute_gradient(function: sp.Expr) -> list[sp.Expr]:
    """Auto-compute gradient for the given function."""
    return cast(
        list[sp.Expr],
        [
            sp.diff(function, var)  # pyright: ignore[reportUnknownMemberType]
            for var in _get_args(function)
        ],
    )


def _autocompute_divergence(function: list[sp.Expr]) -> sp.Expr:
    """Auto-compute divergence for the given function."""
    args = variables[: len(function)]
    x = cast(
        list[sp.Expr],
        [
            sp.diff(func, arg)  # pyright: ignore[reportUnknownMemberType]
            for arg, func in zip(args, function, strict=True)
        ],
    )
    return cast(sp.Expr, sum(x))


def _autocompute_laplace(function: sp.Expr) -> sp.Expr:
    """Auto-compute Laplace for the given function."""
    laplace = cast(
        list[sp.Expr],
        [
            sp.diff(function, var, 2)  # pyright: ignore[reportUnknownMemberType]
            for var in _get_args(function)
        ],
    )
    return cast(sp.Expr, sum(laplace))


def _sample_vector_function(
    function: list[sp.Expr], min_val: float = -1, max_val: float = 1, steps: int = 20
) -> np.ndarray:
    return np.array(
        [
            _sample_function(fce, min_val, max_val, steps, arg_num=len(function))
            for fce in function
        ]
    )


def _sample_function(
    function: sp.Expr,
    min_val: float = -1,
    max_val: float = 1,
    steps: int = 20,
    arg_num: int | None = None,
) -> np.ndarray:
    args = _get_args(function) if arg_num is None else variables[:arg_num]
    if not args:
        return np.array([float(function.evalf())])
    a = np.array(
        [
            _sample_function(
                function.subs(
                    args[-1], i * (max_val - min_val) / (steps - 1) + min_val
                ),
                min_val,
                max_val,
                steps,
                arg_num=len(args) - 1,
            )
            for i in range(steps)
        ]
    )
    print(np.squeeze(a).shape)
    return np.squeeze(a)


SPACIAL_STEP = 0.1

_test_functions = [
    a**2 + b,
]  # a**3, c - 3 + b * a, d * b**2 + a * c]
_test_vector_functions = [
    [a**2 + b, a**3 + b],
    # [c - 3 + b * a + d, d * b**2 + a * c, a * b + b + b + d, 2 + a - d],
]

TEST_TENSORS = [_sample_function(fce) for fce in _test_functions]

TEST_GRADIENTS = [
    _sample_vector_function(_autocompute_gradient(fce)) for fce in _test_functions
]
TEST_VECTOR_TENSORS = [_sample_vector_function(fce) for fce in _test_vector_functions]
TEST_DIVERGENCES = [
    _sample_vector_function(_autocompute_gradient(fce)) for fce in _test_functions
]


@pytest.mark.parametrize(
    ("tensor", "grad"), zip(TEST_TENSORS, TEST_GRADIENTS, strict=True)
)
def test_gradient(tensor: np.ndarray, grad: np.ndarray) -> None:
    """Test numerical gradient is close to the analytical one."""
    got = gradient(tensor, np.array([SPACIAL_STEP] * tensor.ndim))

    def get_interior(A: np.ndarray) -> np.ndarray:
        slices = (slice(None), *tuple(slice(1, -1) for _ in range(A.ndim - 1)))

        return A[slices]

    print(tensor.dtype)
    assert np.allclose(get_interior(got), get_interior(grad), rtol=1e-3)
