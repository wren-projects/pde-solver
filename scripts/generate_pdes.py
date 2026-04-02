# ruff: noqa: ARG001, T201, E501
import inspect
import itertools
from collections.abc import Callable, Iterable
from types import NoneType
from typing import Any, TypeAliasType, cast

import numpy as np

from pde_solver.pde_types import (
    DType,
    Function,
    Matrix,
    MatrixFunction,
    Scalar,
    ScalarFunction,
    TimeFunction,
    Vector,
    VectorFunction,
)

imports = """
from types import NoneType
from typing import Any, Callable

import numpy as np

from pde_solver.pde_types import (
    DType,
    Function,
    Matrix,
    MatrixFunction,
    Scalar,
    ScalarFunction,
    Vector,
    VectorFunction,
)
"""

type data_type = TypeAliasType | type[NoneType]

type Entry = tuple[str, data_type, str]


def to_camel_case(name: str) -> str:
    """Transform a string from snake_case into camel_case."""
    return "".join(word.title() for word in name.split("_"))


def scalar_to_vector(dim: int, value: Scalar) -> Vector:
    """Transform scalar into a vector."""
    return np.full(dim, value, dtype=DType)


def scalar_to_matrix(dim: int, value: Scalar) -> Matrix:
    """Transform scalar into a matrix."""
    return value * np.eye(dim, dtype=DType)


def constant_to_function[T: Scalar | Vector | Matrix](
    dim: int, value: T
) -> Function[T]:
    """Transform scalar into a constant function."""
    return lambda _: value


def constant_zero(dim: int, value: None) -> Scalar:
    """Transform None into zero scalar."""
    return DType(0)


def constant_zero_vector(dim: int, value: None) -> Vector:
    """Transform None into zero vector."""
    return np.zeros(dim, dtype=DType)


def constant_zero_function(dim: int, value: None) -> ScalarFunction:
    """Transform None into zero function."""
    return lambda _: DType(0)


def identity[T](dim: int, value: T) -> T:
    """Transform value into itself."""
    return value


casting: dict[tuple[data_type, data_type], Callable[[int, Any], Any]] = {
    # identity
    (NoneType, NoneType): identity,
    (Matrix, Matrix): identity,
    (Vector, Vector): identity,
    (Scalar, Scalar): identity,
    (ScalarFunction, ScalarFunction): identity,
    (VectorFunction, VectorFunction): identity,
    (MatrixFunction, MatrixFunction): identity,
    (TimeFunction, TimeFunction): identity,
    # None
    (NoneType, Scalar): constant_zero,
    (NoneType, Vector): constant_zero_vector,
    (NoneType, ScalarFunction): constant_zero_function,
    # Scalar
    (Scalar, Matrix): scalar_to_matrix,
    (Scalar, ScalarFunction): constant_to_function,
    # Vector
    (Vector, VectorFunction): constant_to_function,
    # Matrix
    (Matrix, MatrixFunction): constant_to_function,
}

right_side: tuple[Entry, ...] = (
    ("variable_inhomogenity", ScalarFunction, "is some (scalar) function of position."),
    (
        "homogeneous",
        NoneType,
        "is always zero (i.e. homogenoues). Note that the datatype is None.",
    ),
)

advection: tuple[Entry, ...] = (
    (
        "variable_vector_advection",
        VectorFunction,
        "is some (vector) function of position.",
    ),
    ("vector_advection", Vector, "is some constant vector."),
    (
        "no_advection",
        NoneType,
        "is always zero (i.e. there is no advection). Note that the datatype is None.",
    ),
)

diffusion: tuple[Entry, ...] = (
    (
        "variable_matrix_diffusion",
        MatrixFunction,
        "is some (matrix) function of positon.",
    ),
    ("matrix_diffusion", Matrix, "is some constant matrix."),
    (
        "scalar_diffusion",
        Scalar,
        "is some constant scalar. The diffusion is then constant in all directions.",
    ),
    (
        "no_diffusion",
        NoneType,
        "is always zero (i.e. there is no diffusion). Note that the datatype is None.",
    ),
)


def create_name(parts: Iterable[str]) -> str:
    """Create an appropriate class name."""
    return to_camel_case("_".join(parts)) + "PDE"


def create_class_docs(
    class_name: str, right_side: Entry, advection: Entry, diffusion: Entry
) -> str:
    """Create an appropriate docs string for the class."""
    return f'''    """
    {class_name} is a representation of a diffusion-advection PDE.

    The specific form of the PDE is:
    uₜ + ∇(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f

    Where 𝐚 is the vector of advection, 𝚩 is the matrix of diffusion and f is the right hand side.

    In this exact class:
    Diffusion is {diffusion[2]}.
    Advection is {advection[2]}.
    And right hand side {right_side[2]}.
    """
'''


def create_init_docs(
    class_name: str, right_side: Entry, advection: Entry, diffusion: Entry
) -> str:
    """Create an appropriate init doc string for the class."""
    return f'''        """
        Create an implementation of the {class_name} class.

        Parameters
        ----------
        dims: int
            The number of spacial dimensions of the PDE. This notably doesn't include
            the time dimension.

        {right_side[0]}: {right_side[1].__name__}
            The right side of the PDE.
            {right_side[2]}

        {advection[0]}: {advection[1].__name__}
            The advection part of the PDE.
            {advection[2]}

        {diffusion[0]}: {diffusion[1].__name__}
            The diffusion part of the PDE.
            {diffusion[2]}

        """'''


######################################################################
# HERE STOPS THE CODE A FUTURE PROGRAM SHOULD EVER TOUCH OR EVEN SEE #
######################################################################


# IMPORTS AND FUNCTIONS
print("# pyright: reportUnsafeMultipleInheritance=false")
print("# pyright: reportMissingSuperCall=false")
print("# pyright: reportUnusedParameter=false")
print("# pyright: reportAny=false")
print("# ruff: noqa: ARG001, ARG002, E501")
print(imports)
print()

for function in set(casting.values()):
    print(inspect.getsource(function))

#######################################
# I MEAN IT! DON'T LOOK ANY FURTHER! #
#######################################

for (current_right_side, prev_right_side), (
    current_advection,
    prev_advection,
), (current_diffusion, prev_diffusion) in itertools.product(
    zip(right_side, [None, *right_side[:-1]], strict=True),
    zip(advection, [None, *advection[:-1]], strict=True),
    zip(diffusion, [None, *diffusion[:-1]], strict=True),
):
    current_traits = [current_right_side, current_advection, current_diffusion]
    parent_traits = [prev_right_side, prev_advection, prev_diffusion]

    name = create_name(name for name, _, _ in current_traits)
    parent_names: list[str] = []
    super_calls: list[str] = []

    possible_parents = [
        (*current_traits[:i], parent_traits[i], *current_traits[i + 1 :])
        for i in range(len(current_traits))
    ]

    parent_less = True

    for parent_traits in possible_parents:
        if any(trait is None for trait in parent_traits):
            # some part of parent doesn't exist - we are already at the top of the chain
            continue
        parent_less = False

        parent_traits = cast(tuple[Entry, ...], parent_traits)
        parent_name = create_name(name for name, _, _ in parent_traits)
        parent_names.append(parent_name)

        super_init_attributes = ["self", "dims"]
        for current_trait, parent_trait in zip(
            current_traits, parent_traits, strict=True
        ):
            current_trait_name, current_trait_type, _ = current_trait
            parent_trait_name, parent_trait_type, _ = parent_trait

            casted_name = casting[current_trait_type, parent_trait_type].__name__
            super_init_attributes.append(
                f"{parent_trait_name} = {casted_name}(dims, {current_trait_name})"
            )

        super_calls.append(
            f"{parent_name}.__init__({', '.join(super_init_attributes)})"
        )

    arguments = [
        "self",
        "dims: int",
        *(f"{name}: {dtype.__name__}" for name, dtype, _ in current_traits),
    ]
    attributes = [
        string
        for name, dtype, _ in current_traits
        for string in (
            f'self._check_trait(dims, "{name}", {name})',
            f"self.{name}: {dtype.__name__} = {name}",
        )
    ]

    superclasses = f"({', '.join(parent_names)})" if parent_names else ""
    lines: list[str] = []

    ident = "    "
    lines += [
        f"class {name} {superclasses}:",
        create_class_docs(name, *current_traits),
        f"{ident}def __init__({', '.join(arguments)}) -> None:",
        create_init_docs(name, *current_traits),
    ]
    lines += [f"{ident}{ident}{super_call}" for super_call in super_calls]
    lines += [f"{ident}{ident}{att}" for att in attributes]

    if parent_less:
        lines += [
            "",
            f"{ident}def _check_function_equal(self, fce1: Callable, fce2: Callable, dims: int) -> bool:",
            f"{ident}{ident}return np.array_equal(fce1(np.arange(dims)), fce2(np.arange(dims)))",
            "",
            f"{ident}def _check_trait(self, dims:int, name: str, value: Any) -> None:",
            f"{ident}{ident}if not hasattr(self, name):",
            f"{ident}{ident}{ident}return",
            f"{ident}{ident}old = getattr(self, name)",
            f"{ident}{ident}if (old is not value) and (old is None or value is None): return",
            f"{ident}{ident}if (callable(value) and self._check_function_equal(old, value, dims)): return",
            f"{ident}{ident}if (not callable(value)) and ((old is value) or np.array_equal(old, value)): return",
            ident * 2
            + 'raise TypeError(f"PDE structure latice is disrupted! Found value of attribute {name} to be {getattr(self, name)} when it should be {getattr(self, name)}")',
        ]

    print("\n".join(lines))
    print()
    print()
