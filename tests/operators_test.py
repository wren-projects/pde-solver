from typing import cast

import numpy as np
import pytest
import sympy as sp  # pyright: ignore[reportMissingTypeStubs]

from pde_solver.operators import divergence, gradient, laplace
from pde_solver.pde_types import DType, NDArray
from tests.common import assert_default_epsilon, tensor_interior

# TODO: Move
MIN_VAL = 0.5  # we need to not hit any roots of the functions later
MAX_VAL = 10
STEPS = 20
SPACIAL_STEP = (MAX_VAL - MIN_VAL) / (STEPS - 1)
GRID_1D = np.linspace(MIN_VAL, MAX_VAL, STEPS, dtype=float)

VARIABLE_NAMES = ["a", "b", "c", "d", "e", "f"]
A, B, C, D, E, F = VARIABLES = [sp.Symbol(name) for name in VARIABLE_NAMES]


def _infer_args(function: sp.Expr) -> list[sp.Symbol]:
    """
    Find which arguments the function takes.

    To work correctly with functions constant in some arguments, we assume that if a
    function takes f, it takes e, d,… too.
    """
    used_vars = function.free_symbols
    indices = [
        VARIABLE_NAMES.index(str(v)) for v in used_vars if str(v) in VARIABLE_NAMES
    ]
    if not indices:
        return []
    return VARIABLES[: max(indices) + 1]


def _infer_arg_count(function: sp.Expr) -> int:
    return len(_infer_args(function))


def _autocompute_gradient(function: sp.Expr) -> list[sp.Expr]:
    """Auto-compute gradient for the given function."""
    return [
        cast(
            sp.Expr,
            sp.diff(function, var),  # pyright: ignore[reportUnknownMemberType]
        )
        for var in _infer_args(function)
    ]


def _autocompute_divergence(function: list[sp.Expr]) -> sp.Expr:
    """Auto-compute divergence for the given function."""
    args = VARIABLES[: len(function)]
    return cast(
        sp.Expr,
        sum(
            cast(
                sp.Expr,
                sp.diff(func, arg),  # pyright: ignore[reportUnknownMemberType]
            )
            for arg, func in zip(args, function, strict=True)
        ),
    )


def _autocompute_laplace(function: sp.Expr) -> sp.Expr:
    """Auto-compute Laplace for the given function."""
    return cast(
        sp.Expr,
        sum(
            cast(
                sp.Expr,
                sp.diff(function, var, 2),  # pyright: ignore[reportUnknownMemberType]
            )
            for var in _infer_args(function)
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
    args = _infer_args(function) if arg_num is None else VARIABLES[:arg_num]

    grids = np.meshgrid(*([GRID_1D] * len(args)), indexing="ij")

    f = cast(
        sp.FunctionClass,
        sp.lambdify(  # pyright: ignore[reportUnknownMemberType]
            args, function, modules="numpy"
        ),
    )
    out = cast(NDArray, f(*grids))

    # lambdify may return scalar for constant expressions; broadcast to grid shape
    if np.isscalar(out):
        out = np.full([STEPS] * len(args), out, dtype=DType)
    else:
        out = np.asarray(out, dtype=DType)

    return out


TEST_FUNCTIONS = [A**2, (1 + A) * (1 - B), A**3, C - 3 + B * A, D * B**2 + A * C]
TEST_VECTOR_FUNCTIONS = [
    [A**2],
    [A**2 + B**0.5, A**3 + B**2],
    [C - 3 + B * A + D, D * B**2 + A * C, A * B + B + B + D, 2 + A - D],
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
    _sample_function(_autocompute_laplace(fce), arg_num=_infer_arg_count(fce))
    for fce in TEST_FUNCTIONS
]


# we define this function here as it is highly specific and unlikely to be useful
# elsewhere
def get_vector_interior(tensor: np.ndarray, order: int = 1) -> np.ndarray:
    """Return the interior of the tensor (a vector function)."""
    slices = (None, *tuple(slice(order, -order) for _ in range(tensor.ndim - 1)))

    return tensor[slices]


@pytest.mark.parametrize(
    ("tensor", "grad"), zip(TEST_TENSORS, TEST_GRADIENTS, strict=True)
)
def test_gradient(tensor: np.ndarray, grad: np.ndarray) -> None:
    """Test numerical gradient is close to the analytical one."""
    got = gradient(tensor, np.array([SPACIAL_STEP] * tensor.ndim))

    assert_default_epsilon(get_vector_interior(got), get_vector_interior(grad))


@pytest.mark.parametrize(
    ("tensor", "div"), zip(TEST_VECTOR_TENSORS, TEST_DIVERGENCES, strict=True)
)
def test_divergence(tensor: np.ndarray, div: np.ndarray) -> None:
    """Test numerical divergence is close to the analytical one."""
    got = divergence(tensor, np.array([SPACIAL_STEP] * (tensor.ndim - 1)))
    assert_default_epsilon(tensor_interior(got), tensor_interior(div))


@pytest.mark.parametrize(
    ("tensor", "lap"), zip(TEST_TENSORS, TEST_LAPLACES, strict=True)
)
def test_laplace(tensor: np.ndarray, lap: np.ndarray) -> None:
    """Test numerical Laplace is close to the analytical one."""
    got = laplace(tensor, np.array([SPACIAL_STEP] * tensor.ndim))
    # tolerance needs to be different due to high discretization error
    assert_default_epsilon(tensor_interior(got, order=2), tensor_interior(lap, order=2))
