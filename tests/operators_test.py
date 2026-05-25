from typing import cast

import numpy as np
import pytest
import sympy as sp  # pyright: ignore[reportMissingTypeStubs]
from numpy.testing import assert_allclose

from pde_solver.operators import divergence, gradient, laplace
from pde_solver.pde_types import DType, NDArray

# TODO: Move this into a new file
MIN_VAL = 0.5 - 1e-6
MAX_VAL = 0.5 + 1e-6
STEPS = 10
SPACIAL_STEP = (MAX_VAL - MIN_VAL) / (STEPS - 1)

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


def _get_arg_count(function: sp.Expr) -> int:
    return len(_get_args(function))


def _autocompute_gradient(function: sp.Expr) -> list[sp.Expr]:
    """Auto-compute gradient for the given function."""
    return [
        cast(sp.Expr, sp.diff(function, var))  # pyright: ignore[reportUnknownMemberType]
        for var in _get_args(function)
    ]


def _autocompute_divergence(function: list[sp.Expr]) -> sp.Expr:
    """Auto-compute divergence for the given function."""
    args = variables[: len(function)]
    return cast(
        sp.Expr,
        sum(
            cast(sp.Expr, sp.diff(func, arg))  # pyright: ignore[reportUnknownMemberType]
            for arg, func in zip(args, function, strict=True)
        ),
    )


def _autocompute_laplace(function: sp.Expr) -> sp.Expr:
    """Auto-compute Laplace for the given function."""
    return cast(
        sp.Expr,
        sum(
            cast(sp.Expr, sp.diff(function, var, 2))  # pyright: ignore[reportUnknownMemberType]
            for var in _get_args(function)
        ),
    )


def _sample_vector_function(
    function: list[sp.Expr],
) -> np.ndarray:
    return np.array([_sample_function(fce, arg_num=len(function)) for fce in function])


def _sample_function(
    function: sp.Expr,
    arg_num: int | None = None,
) -> np.ndarray:
    args = _get_args(function) if arg_num is None else variables[:arg_num]

    grid_1d = np.linspace(MIN_VAL, MAX_VAL, STEPS, dtype=float)
    grids = np.meshgrid(*([grid_1d] * len(args)), indexing="ij")

    f = cast(
        sp.FunctionClass,
        sp.lambdify(args, function, modules="numpy"),  # pyright: ignore[reportUnknownMemberType]
    )
    out = cast(NDArray, f(*grids))

    # lambdify may return scalar for constant expressions; broadcast to grid shape
    if np.isscalar(out):
        out = np.full([STEPS] * len(args), out, dtype=DType)
    else:
        out = np.asarray(out, dtype=DType)

    return out


TEST_FUNCTIONS = [a**2, (1 + a) * (1 - b), a**3, c - 3 + b * a, d * b**2 + a * c]
TEST_VECTOR_FUNCTIONS = [
    [a**2],
    [a**2 + b**0.5, a**3 + b**2],
    [c - 3 + b * a + d, d * b**2 + a * c, a * b + b + b + d, 2 + a - d],
]

TEST_TENSORS = [_sample_function(fce) for fce in TEST_FUNCTIONS]
TEST_VECTOR_TENSORS = [_sample_vector_function(fce) for fce in TEST_VECTOR_FUNCTIONS]

TEST_GRADIENTS = [
    _sample_vector_function(_autocompute_gradient(fce)) for fce in TEST_FUNCTIONS
]
TEST_DIVERGENCES = [
    _sample_function(_autocompute_divergence(fce), arg_num=len(fce))
    for fce in TEST_VECTOR_FUNCTIONS
]
TEST_LAPLACES = [
    _sample_function(_autocompute_laplace(fce), arg_num=_get_arg_count(fce))
    for fce in TEST_FUNCTIONS
]


def get_vector_interior(tensor: np.ndarray, order: int = 1) -> np.ndarray:
    """Return the interior of the tensor (a vector function)."""
    slices = (None, *tuple(slice(order, -order) for _ in range(tensor.ndim - 1)))

    return tensor[slices]


def get_interior(tensor: np.ndarray, order: int = 1) -> np.ndarray:
    """Return the interior of the tensor by removing the boundary."""
    slices = tuple(slice(order, -order) for _ in range(tensor.ndim))

    return tensor[slices]


@pytest.mark.parametrize(
    ("tensor", "grad"), zip(TEST_TENSORS, TEST_GRADIENTS, strict=True)
)
def test_gradient(tensor: np.ndarray, grad: np.ndarray) -> None:
    """Test numerical gradient is close to the analytical one."""
    got = gradient(tensor, np.array([SPACIAL_STEP] * tensor.ndim))

    assert_allclose(
        get_vector_interior(got), get_vector_interior(grad), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize(
    ("tensor", "div"), zip(TEST_VECTOR_TENSORS, TEST_DIVERGENCES, strict=True)
)
def test_divergence(tensor: np.ndarray, div: np.ndarray) -> None:
    """Test numerical gradient is close to the analytical one."""
    got = divergence(tensor, np.array([SPACIAL_STEP] * (tensor.ndim - 1)))
    assert_allclose(get_interior(got), get_interior(div), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    ("tensor", "lap"), zip(TEST_TENSORS, TEST_LAPLACES, strict=True)
)
def test_laplace(tensor: np.ndarray, lap: np.ndarray) -> None:
    """Test numerical gradient is close to the analytical one."""
    got = laplace(tensor, np.array([SPACIAL_STEP] * tensor.ndim))
    assert_allclose(
        get_interior(got, order=2), get_interior(lap, order=2), rtol=1e-3, atol=1e-1
    )
