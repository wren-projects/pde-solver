# ruff: noqa: ARG001, N999, T201
from enum import Enum, StrEnum
import inspect
import itertools
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeAliasType, cast
from types import NoneType

import numpy as np

from pde_solver.our_types import DType, Function, Matrix, Scalar, TimeFunction, Vector


type data_type = TypeAliasType | type[NoneType]

type Entry = tuple[str, data_type]


def to_camel_case(name: str) -> str:
    """Transform a string from snake_case into camle_case."""
    return "".join(word.title() for word in name.split())


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


def constant_zero_function(dim: int, value: None) -> Function[Scalar]:
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
    (Function, Function): identity,
    (TimeFunction, TimeFunction): identity,
    # None
    (NoneType, Scalar): constant_zero,
    (NoneType, Vector): constant_zero_vector,
    (NoneType, Function): constant_zero_function,
    # Scalar
    (Scalar, Matrix): scalar_to_matrix,
    (Scalar, Function): constant_to_function,
    # Vector
    (Vector, Function): constant_to_function,
    # Matrix
    (Matrix, Function): constant_to_function,
}

right_side: tuple[Entry, ...] = (
    ("VariableInhomogenity", Function),
    ("Homogeneous", NoneType),
)

advection: tuple[Entry, ...] = (
    ("VariableVectorAdvection", Function),
    ("VectorAdvection", Vector),
    ("NoAdvection", NoneType),
)

diffusion: tuple[Entry, ...] = (
    ("VariableMatrixDiffusion", Function),
    ("MatrixDiffusion", Matrix),
    ("ScalarDiffusion", Scalar),
    ("NoDiffusion", NoneType),
)


def create_name(parts: Iterable[str]) -> str:
    """Create an appropriate class name."""
    return "".join(parts) + "PDE"


# IMPORTS AND FUNCTIONS
print("# pyright: reportUnsafeMultipleInheritance=false")
print("# pyright reportMissingSuperCall=false")
print(
    "".join(
        [
            line
            for line in Path(__file__).read_text()
            if line.startswith(("from ", "import "))
        ]
    )
)

for function in set(casting.values()):
    print(inspect.getsource(function))


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

    name = create_name(name for name, _ in current_traits)
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
        parent_name = create_name(name for name, _ in parent_traits)
        parent_names.append(parent_name)

        super_init_attributes = ["self", "dims"]
        for current_trait, parent_trait in zip(
            current_traits, parent_traits, strict=True
        ):
            current_trait_name, current_trait_type = current_trait
            parent_trait_name, parent_trait_type = parent_trait

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
        *(f"{name}: {dtype.__name__}" for name, dtype in current_traits),
    ]
    attributes = [
        string
        for name, _ in current_traits
        for string in (f"self._set_trait('{name}', {name})", f"self.{name} = {name}")
    ]
    attributes = [f"self._set_trait('{name}', {name})" for name, _ in current_traits]

    print(
        f"""
class {name} ({", ".join(parent_names)}):
    def __init__({", ".join(arguments)}):
        {"\n        ".join(super_calls)}
        {"\n        ".join(attributes)}"""
    )

    if parent_less:
        print("""
    def _set_trait(self, name: str, value: str):
        if hasattr(self, name) and getattr(self, name) != value:
            raise TypeError(
                r"PDE structure latice is disrupted! Found value of attribute"
                f"{name} to be {getattr(self, name)} when it should be"
                f"{getattr(self, value)}"
            )

        setattr(self, name, value)
        """)
