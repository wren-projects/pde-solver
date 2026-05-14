from collections.abc import Callable
from itertools import combinations
from typing import cast

import numpy as np
import pytest

from pde_solver.operators import divergence, gradient, laplace
from pde_solver.pde_types import DType, NDArray, Vector

DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-6

CONSTANT_CASES = [
    ((5, 6), np.array([0.5] * 2, dtype=DType), 3.5),
    ((7, 4, 6), np.array([0.3, 0.4, 1.2], dtype=DType), -1.1),
    (
        (5, 4, 6, 3, 2),
        np.array([0.8] * 5, dtype=DType),
        0.0,
    ),
]

LINEAR_CASES = [
    (
        (4, 5, 6),
        np.array([0.5] * 3, dtype=DType),
        np.array([2.0, -1.5, 0.25], dtype=DType),
        4.2,
    ),
    (
        (6, 4),
        np.array([0.2, 0.9], dtype=DType),
        np.array([1.1, 0.7], dtype=DType),
        -2.0,
    ),
    (
        (5, 4, 6, 3, 2),
        np.array([0.2, 0.5, 0.7, 1.1, 0.9], dtype=DType),
        np.array([1.1, -0.7, 0.4, 2.1, -0.3], dtype=DType),
        0.3,
    ),
]

QUADRATIC_CASES = [
    (
        (7, 6, 5, 4),
        np.array([0.1] * 4, dtype=DType),
        np.array([1.3, -0.7, 0.4, 2.1], dtype=DType),
    ),
    (
        (6, 5, 4),
        np.array([0.1] * 3, dtype=DType),
        np.array([-0.6, 1.1, 0.9], dtype=DType),
    ),
    (
        (5, 4, 6, 3, 2),
        np.array([0.1] * 5, dtype=DType),
        np.array([1.1, -0.7, 0.4, 2.1, -0.3], dtype=DType),
    ),
]

DIV_CONSTANT_CASES = [
    ((6, 5), np.array([0.5] * 2, dtype=DType), (1.2, -3.4)),
    ((4, 6, 5), np.array([0.3, 0.9, 1.1], dtype=DType), (2.0, -1.0, 0.5)),
    (
        (5, 4, 6, 3, 2),
        np.array([1.0] * 5, dtype=DType),
        (-1.1, 0.7, -0.4, 2.1, 0.3),
    ),
]

DIV_LINEAR_CASES = [
    (
        (5, 6, 4),
        np.array([0.5, 0.3, 1.1], dtype=DType),
        np.array(
            [
                [1.2, 0.3, -0.2],
                [-0.5, 2.5, 0.1],
                [0.0, -0.7, 0.9],
            ],
            dtype=DType,
        ),
        1.5,
    ),
    (
        (6, 4),
        np.array([0.6] * 2, dtype=DType),
        np.array([[1.1, -0.4], [0.2, -2.3]], dtype=DType),
        -0.7,
    ),
    (
        (5, 4, 6, 3, 2),
        np.array([0.2, 0.5, 0.7, 1.1, 0.9], dtype=DType),
        np.array(
            [
                [1.1, -0.4, 0.2, 0.5, -0.3],
                [0.2, 2.3, -0.1, 0.4, 0.7],
                [0.0, 0.6, 1.5, -0.2, 0.1],
                [0.3, -0.5, 0.8, 2.1, -0.4],
                [-0.7, 0.2, -0.6, 0.9, 1.3],
            ],
            dtype=DType,
        ),
        0.2,
    ),
]


LAPLACE_BILINEAR_CASES = [
    ((7, 6), np.array([0.6] * 2, dtype=DType), 2.5),
    ((8, 5), np.array([0.4, 1.1], dtype=DType), -1.7),
]

CROSS_QUADRATIC_CASES = [
    (
        (6, 5),
        np.array([0.1] * 2, dtype=DType),
        {
            "linear": np.array([1.2, -0.7], dtype=DType),
            "quadratic": np.array([0.9, -0.4], dtype=DType),
            "cross": np.array([0.8], dtype=DType),
            "constant": 1.3,
        },
    ),
    (
        (5, 4, 6),
        np.array([0.1] * 3, dtype=DType),
        {
            "linear": np.array([0.8, -1.1, 0.6], dtype=DType),
            "quadratic": np.array([0.5, -0.2, 1.4], dtype=DType),
            "cross": np.array([0.7, -0.3, 0.2], dtype=DType),
            "constant": -0.5,
        },
    ),
    (
        (5, 4, 6, 3),
        np.array([0.1] * 4, dtype=DType),
        {
            "linear": np.array([0.6, -0.4, 0.3, 0.2], dtype=DType),
            "quadratic": np.array([0.2, -0.1, 0.4, 0.3], dtype=DType),
            "cross": np.array([0.5, -0.3, 0.2, -0.4, 0.1, 0.6], dtype=DType),
            "constant": 0.7,
        },
    ),
]

BILINEAR_VECTOR_CASES = [
    (
        (6, 5),
        np.array([0.5] * 2, dtype=DType),
        (1.1, -0.6),
    ),
    (
        (5, 4, 6),
        np.array([0.3, 0.4, 0.9], dtype=DType),
        (0.8, -1.2, 0.5),
    ),
]

NONPOLY_SCALAR_CASES = [
    (
        (61, 60),
        np.array([0.0001] * 2, dtype=DType),
        np.array([0.6, -0.4], dtype=DType),
        0.3,
    ),
    (
        (40, 36, 32),
        np.array([0.001, 0.0012, 0.0008], dtype=DType),
        np.array([0.5, -0.3, 0.4], dtype=DType),
        -0.2,
    ),
]

NONPOLY_VECTOR_CASES = [
    (
        (60, 54),
        np.array([0.01] * 2, dtype=DType),
        np.array([0.25, -0.2], dtype=DType),
    ),
    (
        (36, 33, 30),
        np.array([0.01, 0.012, 0.008], dtype=DType),
        np.array([0.2, -0.15, 0.25], dtype=DType),
    ),
]


def make_grid(shape: tuple[int, ...], steps: Vector) -> NDArray:
    """Create coordinate grids for each axis."""
    axes = [
        np.arange(size, dtype=float) * step
        for size, step in zip(shape, steps, strict=True)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"))


def linear_field(coords: NDArray, coeffs: NDArray, offset: float = 0.0) -> NDArray:
    """Build sumᵢ coeffs[i] * coords[i]."""
    return np.tensordot(coeffs, coords, axes=1) + offset


def quadratic_field(coords: NDArray, coeffs: NDArray) -> NDArray:
    """Build sumᵢ coeffs[i] * coords[i]**2."""
    return linear_field(coords**2, coeffs)


def cubic_field(coords: NDArray, coeffs: NDArray) -> NDArray:
    """Build sumᵢ coeffs[i] * coords[i]**3."""
    return linear_field(coords**3, coeffs)


def product_except(coords: NDArray, index: int) -> NDArray:
    """Return product of all coordinate axes except one."""
    factors = [coords[i] for i in range(len(coords)) if i != index]
    return np.asarray(np.prod(factors, axis=0), dtype=DType)


def cross_quadratic_field(
    coords: NDArray,
    linear_coeffs: Vector,
    quadratic_coeffs: Vector,
    cross_coeffs: Vector,
    constant: float,
) -> NDArray:
    """Build linear + quadratic + pairwise cross terms."""
    pairs = combinations(range(len(coords)), 2)
    cross = sum(
        coeff * coords[i] * coords[j] for coeff, (i, j) in zip(cross_coeffs, pairs)
    )
    return (
        linear_field(coords, linear_coeffs, constant)
        + quadratic_field(coords, quadratic_coeffs)
        + cross
    )


def cross_quadratic_gradient(
    coords: NDArray,
    linear_coeffs: Vector,
    quadratic_coeffs: Vector,
    cross_coeffs: Vector,
) -> NDArray:
    """Build analytic gradient of linear + quadratic + pairwise cross terms."""
    grads: list[NDArray] = (linear_coeffs + 2.0 * quadratic_coeffs * coords.T).T
    pairs = combinations(range(len(coords)), 2)
    for coeff, (i, j) in zip(cross_coeffs, pairs):
        grads[i] = grads[i] + coeff * coords[j]
        grads[j] = grads[j] + coeff * coords[i]
    return np.stack(grads)


def interior_slices(ndim: int, width: int = 1) -> tuple[slice, ...]:
    """Create slices for a centered interior region."""
    return tuple(slice(width, -width) for _ in range(ndim))


@pytest.mark.parametrize(
    ("shape", "steps", "value"),
    CONSTANT_CASES,
)
def test_gradient_constant_field_is_zero(
    shape: tuple[int, ...], steps: Vector, value: float
) -> None:
    """Gradient of a constant field is zero."""
    tensor = np.full(shape, value)
    grads = gradient(tensor, steps)
    np.testing.assert_allclose(grads, 0, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs", "offset"),
    LINEAR_CASES,
)
def test_gradient_linear_field_is_constant(
    shape: tuple[int, ...],
    steps: Vector,
    coeffs: Vector,
    offset: float,
) -> None:
    """Gradient of a linear field is constant everywhere."""
    coords = make_grid(shape, steps)
    tensor = linear_field(coords, coeffs, offset)
    grads = gradient(tensor, steps)
    expected = np.broadcast_to(coeffs[:, *([np.newaxis] * len(shape))], grads.shape)
    np.testing.assert_allclose(grads, expected, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs"),
    QUADRATIC_CASES,
)
def test_gradient_quadratic_field_matches_analytic(
    shape: tuple[int, ...], steps: Vector, coeffs: Vector
) -> None:
    """
    Gradient of a quadratic field matches analytic derivative.

        f(x) = Σᵢ cᵢ xᵢ²
        ∂ᵢf = 2 cᵢ xᵢ

    """
    coords = make_grid(shape, steps)
    tensor = quadratic_field(coords, coeffs)
    grads = gradient(tensor, steps)
    expected = 2.0 * coeffs[:, *([np.newaxis] * len(shape))] * coords
    interior = interior_slices(len(shape))
    np.testing.assert_allclose(
        grads[:, *interior],
        expected[:, *interior],
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


@pytest.mark.parametrize(
    ("shape", "steps", "values"),
    CONSTANT_CASES,
)
def test_divergence_constant_vector_field_is_zero(
    shape: tuple[int, ...], steps: Vector, values: float
) -> None:
    """
    Divergence of a constant vector field is zero.

        f(x) = c
        ∇⋅f = 0

    """
    vector_field = np.stack([np.full(shape, values) for _ in range(len(shape))])
    div = divergence(vector_field, steps)
    np.testing.assert_allclose(div, 0, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


@pytest.mark.parametrize(
    ("shape", "steps", "matrix", "offset"),
    DIV_LINEAR_CASES,
)
def test_divergence_linear_vector_field_is_trace(
    shape: tuple[int, ...],
    steps: Vector,
    matrix: NDArray,
    offset: float,
) -> None:
    """
    Divergence of a linear vector field equals trace.

        f(x) = Σᵢ aᵢ xᵢ
        ∇⋅f = Σᵢ aᵢ

    """
    coords = make_grid(shape, steps)
    vector_field = linear_field(coords, matrix, offset)
    np.testing.assert_allclose(
        divergence(vector_field, steps),
        np.trace(matrix),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs"),
    QUADRATIC_CASES,
)
def test_divergence_quadratic_vector_field_matches_analytic(
    shape: tuple[int, ...], steps: Vector, coeffs: NDArray
) -> None:
    """
    Divergence of quadratic components matches analytic result.

        f(x) = Σᵢ aᵢ xᵢ²
        ∇⋅f = 2 Σᵢ aᵢ xᵢ

    """
    coords = make_grid(shape, steps)
    vector_field: NDArray = coeffs[:, *([np.newaxis] * len(shape))] * (coords**2)
    div = divergence(vector_field, steps)
    expected = linear_field(coords, 2.0 * coeffs)
    interior = interior_slices(len(shape))
    np.testing.assert_allclose(
        div[interior], expected[interior], rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs", "constant"),
    LINEAR_CASES,
)
def test_laplace_constant_and_linear_fields_are_zero(
    shape: tuple[int, ...],
    steps: Vector,
    coeffs: NDArray,
    constant: float,
) -> None:
    """
    Laplacian of constant and linear fields is zero.

        f(x) = c + Σᵢ aᵢ xᵢ
        ∇² f = 0

    """
    coords = make_grid(shape, steps)
    constant_field = np.full(shape, constant)
    linear = linear_field(coords, coeffs, constant)
    np.testing.assert_allclose(
        laplace(constant_field, steps),
        0,
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )
    np.testing.assert_allclose(
        laplace(linear, steps), 0, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


@pytest.mark.parametrize(
    ("shape", "steps", "scale"),
    LAPLACE_BILINEAR_CASES,
)
def test_laplace_bilinear_field_is_zero(
    shape: tuple[int, ...], steps: Vector, scale: float
) -> None:
    """
    Laplacian of a bilinear field is zero in the interior.

        f(x) = Σᵢ aᵢ xᵢ ̄̄̅x̄ᵢ, x̄ᵢ = Πⱼ xᵢ, i ≠ j
        ∇²f = 0

    """
    coords = make_grid(shape, steps)
    tensor = scale * product_except(coords, len(shape) - 1)
    lap = laplace(tensor, steps)
    np.testing.assert_allclose(
        lap,
        0,
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs"),
    QUADRATIC_CASES,
)
def test_laplace_quadratic_field_is_constant(
    shape: tuple[int, ...], steps: Vector, coeffs: NDArray
) -> None:
    """
    Laplacian of a quadratic field is constant in the interior.

        f(x) = Σᵢ cᵢ xᵢ²
        ∇²f = 2 Σᵢ cᵢ

    """
    coords_stack = make_grid(shape, steps)
    tensor = quadratic_field(coords_stack, coeffs)
    lap = laplace(tensor, steps)
    interior = interior_slices(len(shape), width=2)
    np.testing.assert_allclose(
        lap[interior], 2.0 * np.sum(coeffs), rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs"),
    CROSS_QUADRATIC_CASES,
)
def test_gradient_cross_quadratic_field_matches_analytic(
    shape: tuple[int, ...], steps: Vector, coeffs: dict[str, Vector | float]
) -> None:
    """
    Gradient of a cross-quadratic field matches analytic derivative.

        f

    """
    coords = make_grid(shape, steps)
    linear_coeffs = cast(Vector, coeffs["linear"])
    quadratic_coeffs = cast(Vector, coeffs["quadratic"])
    cross_coeffs = cast(Vector, coeffs["cross"])
    constant = cast(float, coeffs["constant"])
    tensor: NDArray = cross_quadratic_field(
        coords,
        linear_coeffs,
        quadratic_coeffs,
        cross_coeffs,
        constant,
    )
    grads = gradient(tensor, steps)
    expected = cross_quadratic_gradient(
        coords, linear_coeffs, quadratic_coeffs, cross_coeffs
    )
    interior = interior_slices(len(shape))
    np.testing.assert_allclose(
        grads[:, *interior],
        expected[:, *interior],
        rtol=7e-2,
        atol=7e-2,
    )


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs"),
    BILINEAR_VECTOR_CASES,
)
def test_divergence_bilinear_vector_field_matches_analytic(
    shape: tuple[int, ...], steps: Vector, coeffs: Vector
) -> None:
    """
    Divergence of a bilinear vector field matches analytic result.

        f(x) = a Πᵢ xᵢ
        ∇⋅f = ∑ᵢ aᵢ Πⱼ xⱼ, i ≠ j

    """
    coords = make_grid(shape, steps)

    product: NDArray = np.prod(coords, axis=0)
    vector_field: NDArray = np.stack([coeff * product for coeff in coeffs])
    expected: NDArray = sum(
        coeff * product_except(coords, i) for i, coeff in enumerate(coeffs)
    )
    div = divergence(vector_field, steps)
    np.testing.assert_allclose(div, expected, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs"),
    CROSS_QUADRATIC_CASES,
)
def test_laplace_cross_quadratic_field_is_constant(
    shape: tuple[int, ...], steps: Vector, coeffs: dict[str, NDArray | float]
) -> None:
    """
    Laplacian of a cross-quadratic field is constant in the interior.

        f(x) = Σᵢ aᵢ xᵢ² + Σᵢ bᵢ xᵢ + c
        ∇²f = 2 Σᵢ aᵢ + 2 Σᵢ bᵢ

    """
    linear_coeffs = cast(NDArray, coeffs["linear"])
    quadratic_coeffs = cast(NDArray, coeffs["quadratic"])
    cross_coeffs = cast(NDArray, coeffs["cross"])
    constant = cast(float, coeffs["constant"])

    coords = make_grid(shape, steps)
    tensor: NDArray = cross_quadratic_field(
        coords,
        linear_coeffs,
        quadratic_coeffs,
        cross_coeffs,
        constant,
    )
    lap = laplace(tensor, steps)
    interior = interior_slices(len(shape), width=2)
    np.testing.assert_allclose(
        lap[interior],
        2.0 * np.sum(quadratic_coeffs),
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs", "offset"),
    NONPOLY_SCALAR_CASES,
)
def test_gradient_sine_field_matches_analytic(
    shape: tuple[int, ...],
    steps: Vector,
    coeffs: NDArray,
    offset: float,
) -> None:
    """
    Gradient of a sine field matches analytic derivative in the interior.

        f(x) = sin(Σᵢ aᵢ xᵢ)
        ∂ᵢ f = aᵢ cos(Σᵢ aᵢ xᵢ)

    """
    # Use finer grids for non-polynomial fields to reduce finite-difference error
    # and keep the global tolerances consistent across all tests.
    coords = make_grid(shape, steps)
    phases = linear_field(coords, coeffs, offset)
    tensor = np.sin(phases)
    grads = gradient(tensor, steps)
    expected = np.stack([coeff * np.cos(phases) for coeff in coeffs])
    interior = interior_slices(len(shape))
    np.testing.assert_allclose(
        grads[:, *interior],
        expected[:, *interior],
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs", "offset"),
    LINEAR_CASES,
)
def test_laplace_sine_field_matches_analytic(
    shape: tuple[int, ...],
    steps: Vector,
    coeffs: NDArray,
    offset: float,
) -> None:
    """
    Laplacian of a sine field matches analytic derivative in the interior.

        f(x) = sin(Σᵢ aᵢ xᵢ)
        ∇²f = - (Σᵢ aᵢ²) sin(Σᵢ aᵢ xᵢ)

    """
    # Use finer grids for non-polynomial fields to reduce finite-difference error
    # and keep the global tolerances consistent across all tests.
    coords = make_grid(shape, steps)
    phases = linear_field(coords, coeffs, offset)
    tensor = np.sin(phases)
    lap = laplace(tensor, steps)
    expected = -np.sum(coeffs**2) * np.sin(phases)
    interior = interior_slices(len(shape), width=2)
    np.testing.assert_allclose(
        lap[interior], expected[interior], rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs", "offset"),
    NONPOLY_SCALAR_CASES,
)
def test_divergence_sin_vector_field_matches_analytic(
    shape: tuple[int, ...], steps: Vector, coeffs: NDArray, offset: float
) -> None:
    """
    Divergence of separable sine vector fields matches analytic result.

        Fᵢ(x) = sin(aᵢ xᵢ + c)
        ∇⋅F = Σᵢ aᵢ cos(aᵢ xᵢ + c)

    """
    # Use finer grids for non-polynomial fields to reduce finite-difference error
    # and keep the global tolerances consistent across all tests.
    coords = make_grid(shape, steps)
    phases = linear_field(coords, coeffs, offset)
    vector_field = np.stack([np.sin(phases)] * len(coeffs))
    div = divergence(vector_field, steps)
    expected = sum(coeff * np.cos(phases) for coeff in coeffs)
    interior = interior_slices(len(shape))
    np.testing.assert_allclose(
        div[interior], expected[interior], rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


@pytest.mark.parametrize(
    ("shape", "steps", "coeffs", "offset"),
    NONPOLY_SCALAR_CASES,
)
def test_divergence_exp_vector_field_matches_analytic(
    shape: tuple[int, ...], steps: Vector, coeffs: Vector, offset: float
) -> None:
    """
    Divergence of exp-scaled vector fields matches analytic result.

        Fᵢ(x) = aᵢ exp(Σⱼ aⱼ xⱼ)
        ∇⋅F = (Σᵢ aᵢ²) exp(Σⱼ aⱼ xⱼ)

    """
    # Use finer grids for non-polynomial fields to reduce finite-difference error
    # and keep the global tolerances consistent across all tests.
    coords = make_grid(shape, steps)
    phases = linear_field(coords, coeffs) + offset
    vector_field = np.stack([coeff * np.exp(phases) for coeff in coeffs])
    div = divergence(vector_field, steps)
    expected = np.sum(coeffs**2) * np.exp(phases)
    interior = interior_slices(len(shape))
    np.testing.assert_allclose(
        div[interior], expected[interior], rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
    )


def assert_linearity(
    op: Callable[[NDArray, Vector], NDArray],
    case: tuple[tuple[int, ...], Vector, float, float],
    seed: int,
    *,
    vector_field: bool,
) -> None:
    """Assert linearity of an operator for random inputs."""
    shape, steps, alpha, beta = case
    rng = np.random.default_rng(seed)
    if vector_field:
        a = rng.normal(size=(len(shape), *shape))
        b = rng.normal(size=(len(shape), *shape))
    else:
        a = rng.normal(size=shape)
        b = rng.normal(size=shape)
    lhs = op(alpha * a + beta * b, steps)
    rhs = np.add(alpha * op(a, steps), beta * op(b, steps))
    np.testing.assert_allclose(lhs, rhs, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


LINEARITY_CASES = [
    (
        (6, 5, 4),
        np.array([0.5, 0.7, 1.2], dtype=DType),
        1.4,
        -0.6,
    ),
    (
        (5, 6),
        np.array([0.3, 0.9], dtype=DType),
        -0.8,
        2.3,
    ),
    (
        (7, 4),
        np.array([0.2, 0.8], dtype=DType),
        0.75,
        -1.25,
    ),
]


@pytest.mark.parametrize(
    ("shape", "steps", "alpha", "beta"),
    LINEARITY_CASES,
)
def test_gradient_is_linear(
    shape: tuple[int, ...], steps: Vector, alpha: float, beta: float
) -> None:
    """Gradient is linear for random fields."""
    assert_linearity(
        gradient,
        (shape, steps, alpha, beta),
        seed=0,
        vector_field=False,
    )


@pytest.mark.parametrize(
    ("shape", "steps", "alpha", "beta"),
    LINEARITY_CASES,
)
def test_divergence_is_linear(
    shape: tuple[int, ...], steps: Vector, alpha: float, beta: float
) -> None:
    """Divergence is linear for random vector fields."""
    assert_linearity(
        divergence,
        (shape, steps, alpha, beta),
        seed=0,
        vector_field=True,
    )


@pytest.mark.parametrize(
    ("shape", "steps", "alpha", "beta"),
    LINEARITY_CASES,
)
def test_laplace_is_linear(
    shape: tuple[int, ...], steps: Vector, alpha: float, beta: float
) -> None:
    """Laplacian is linear for random fields."""
    assert_linearity(
        laplace,
        (shape, steps, alpha, beta),
        seed=0,
        vector_field=False,
    )
