"""
Type-safe incremental builder for PDE solver invocations.

The :class:`SolutionBuilder` collects the seven inputs a
:class:`~wren_pde_solver.abc.solver.Solver` needs (PDE, solver, initial
condition, spatial step, boundary condition, time step and target time) one
at a time. Each ``with_*`` method returns a *new* builder whose type
parameters record which fields are already set, so :meth:`SolutionBuilder.compute`
is only available once every field holds a real value (see :data:`Complete`).

PDE/solver compatibility is enforced statically: ``with_pde`` and
``with_solver`` only accept combinations where the PDE is a subtype of the
solver's capability (``Solver`` is contravariant in its PDE parameter), or
where the counterpart is still unset. Incompatible combinations are type
errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final, overload

from wren_pde_solver.abc.boundary import BoundaryCondition
from wren_pde_solver.abc.pde import PDE
from wren_pde_solver.abc.solver import Solver
from wren_pde_solver.pde_types import DType, NDArray, Vector


@final
@dataclass(frozen=True, slots=True)
class PdeUnset:
    """Sentinel marking that no PDE has been set on the builder yet."""


@final
@dataclass(frozen=True, slots=True)
class SolverUnset:
    """Sentinel marking that no solver has been set on the builder yet."""


@final
@dataclass(frozen=True, slots=True)
class InitialConditionUnset:
    """Sentinel marking that no initial condition has been set yet."""


@final
@dataclass(frozen=True, slots=True)
class SpatialStepUnset:
    """Sentinel marking that no spatial step has been set yet."""


@final
@dataclass(frozen=True, slots=True)
class BoundaryConditionUnset:
    """Sentinel marking that no boundary condition has been set yet."""


@final
@dataclass(frozen=True, slots=True)
class TimeStepUnset:
    """Sentinel marking that no timestep has been set yet."""


@final
@dataclass(frozen=True, slots=True)
class TargetTimeUnset:
    """Sentinel marking that no target time has been set yet."""


PDE_UNSET = PdeUnset()
SOLVER_UNSET = SolverUnset()
INITIAL_CONDITION_UNSET = InitialConditionUnset()
SPATIAL_STEP_UNSET = SpatialStepUnset()
BOUNDARY_CONDITION_UNSET = BoundaryConditionUnset()
TIME_STEP_UNSET = TimeStepUnset()
TARGET_TIME_UNSET = TargetTimeUnset()


@final
@dataclass(frozen=True, slots=True)
class SolutionBuilder[
    PdeT: PDE | PdeUnset = PdeUnset,
    SolverT: Solver[Any] | SolverUnset = SolverUnset,
    InitialConditionT: NDArray | InitialConditionUnset = InitialConditionUnset,
    SpatialStepT: Vector | SpatialStepUnset = SpatialStepUnset,
    BoundaryConditionT: BoundaryCondition | BoundaryConditionUnset = (
        BoundaryConditionUnset
    ),
    TimeStepT: DType | TimeStepUnset = TimeStepUnset,
    TargetTimeT: DType | TargetTimeUnset = TargetTimeUnset,
]:
    """
    Immutable builder accumulating the inputs of a solver call.

    Type parameters track, per field, whether a real value or the
    corresponding ``*Unset`` sentinel is stored. Every ``with_*`` method
    returns a new instance with only its own type parameter updated, except
    ``with_pde``/``with_solver`` which additionally enforce PDE/solver
    compatibility (see module docstring).

    Attributes
    ----------
    pde : PdeT
        The PDE to solve, or :data:`PDE_UNSET` if not set yet.
    solver : SolverT
        The solver to use, or :data:`SOLVER_UNSET` if not set yet.
    initial_condition : InitialConditionT
        Discretized initial state, or :data:`INITIAL_CONDITION_UNSET`.
    spatial_step : SpatialStepT
        Discretization steps per axis, or :data:`SPATIAL_STEP_UNSET`.
    boundary_condition : BoundaryConditionT
        Boundary condition, or :data:`BOUNDARY_CONDITION_UNSET`.
    time_step : TimeStepT
        Solver time increment, or :data:`TIME_STEP_UNSET`.
    target_time : TargetTimeT
        Time at which the solution is requested, or :data:`TARGET_TIME_UNSET`.

    """

    pde: PdeT
    solver: SolverT
    initial_condition: InitialConditionT
    spatial_step: SpatialStepT
    boundary_condition: BoundaryConditionT
    time_step: TimeStepT
    target_time: TargetTimeT

    @overload
    def with_pde[P: PDE](
        self: SolutionBuilder[
            PdeT,
            Solver[P],
            InitialConditionT,
            SpatialStepT,
            BoundaryConditionT,
            TimeStepT,
            TargetTimeT,
        ],
        pde: P,
        /,
    ) -> SolutionBuilder[
        P,
        SolverT,
        InitialConditionT,
        SpatialStepT,
        BoundaryConditionT,
        TimeStepT,
        TargetTimeT,
    ]: ...

    @overload
    def with_pde[P: PDE](
        self: SolutionBuilder[
            PdeT,
            SolverUnset,
            InitialConditionT,
            SpatialStepT,
            BoundaryConditionT,
            TimeStepT,
            TargetTimeT,
        ],
        pde: P,
        /,
    ) -> SolutionBuilder[
        P,
        SolverT,
        InitialConditionT,
        SpatialStepT,
        BoundaryConditionT,
        TimeStepT,
        TargetTimeT,
    ]: ...

    def with_pde[P: PDE](
        self, pde: P, /
    ) -> SolutionBuilder[
        P,
        SolverT,
        InitialConditionT,
        SpatialStepT,
        BoundaryConditionT,
        TimeStepT,
        TargetTimeT,
    ]:
        """
        Return a copy of this builder with ``pde`` replaced.

        Parameters
        ----------
        pde : P
            The PDE to solve from now on. If a solver is already set,
            ``pde`` must be compatible with it (a subtype of the solver's
            PDE parameter); otherwise this call is a static type error.

        Returns
        -------
        SolutionBuilder
            A new builder holding ``pde`` and all other fields unchanged.

        """
        return SolutionBuilder(
            pde,
            self.solver,
            self.initial_condition,
            self.spatial_step,
            self.boundary_condition,
            self.time_step,
            self.target_time,
        )

    @overload
    def with_solver[P: PDE](
        self: SolutionBuilder[
            P,
            SolverT,
            InitialConditionT,
            SpatialStepT,
            BoundaryConditionT,
            TimeStepT,
            TargetTimeT,
        ],
        solver: Solver[P],
        /,
    ) -> SolutionBuilder[
        PdeT,
        Solver[P],
        InitialConditionT,
        SpatialStepT,
        BoundaryConditionT,
        TimeStepT,
        TargetTimeT,
    ]: ...

    @overload
    def with_solver[P: PDE](
        self: SolutionBuilder[
            PdeUnset,
            SolverT,
            InitialConditionT,
            SpatialStepT,
            BoundaryConditionT,
            TimeStepT,
            TargetTimeT,
        ],
        solver: Solver[P],
        /,
    ) -> SolutionBuilder[
        PdeT,
        Solver[P],
        InitialConditionT,
        SpatialStepT,
        BoundaryConditionT,
        TimeStepT,
        TargetTimeT,
    ]: ...

    def with_solver[P: PDE](
        self, solver: Solver[P], /
    ) -> SolutionBuilder[
        PdeT,
        Solver[P],
        InitialConditionT,
        SpatialStepT,
        BoundaryConditionT,
        TimeStepT,
        TargetTimeT,
    ]:
        """
        Return a copy of this builder with ``solver`` replaced.

        Parameters
        ----------
        solver : Solver[P]
            The solver to use from now on.  If a PDE is already set, it
            must be compatible with ``solver`` (a subtype of ``P``);
            otherwise this call is a static type error.

        Returns
        -------
        SolutionBuilder
            A new builder holding ``solver`` and all other fields unchanged.

        """
        return SolutionBuilder(
            self.pde,
            solver,
            self.initial_condition,
            self.spatial_step,
            self.boundary_condition,
            self.time_step,
            self.target_time,
        )

    def with_initial_condition(
        self, initial_condition: NDArray, /
    ) -> SolutionBuilder[
        PdeT,
        SolverT,
        NDArray,
        SpatialStepT,
        BoundaryConditionT,
        TimeStepT,
        TargetTimeT,
    ]:
        """
        Return a copy of this builder with the initial condition replaced.

        Parameters
        ----------
        initial_condition : NDArray
            The already-discretized initial state of the PDE.

        Returns
        -------
        SolutionBuilder
            A new builder holding ``initial_condition``.

        """
        return SolutionBuilder(
            self.pde,
            self.solver,
            initial_condition,
            self.spatial_step,
            self.boundary_condition,
            self.time_step,
            self.target_time,
        )

    def with_spatial_step(
        self, spatial_step: Vector, /
    ) -> SolutionBuilder[
        PdeT,
        SolverT,
        InitialConditionT,
        Vector,
        BoundaryConditionT,
        TimeStepT,
        TargetTimeT,
    ]:
        """
        Return a copy of this builder with the spatial step replaced.

        Parameters
        ----------
        spatial_step : Vector
            Per-axis discretization steps matching the initial condition.

        Returns
        -------
        SolutionBuilder
            A new builder holding ``spatial_step``.

        """
        return SolutionBuilder(
            self.pde,
            self.solver,
            self.initial_condition,
            spatial_step,
            self.boundary_condition,
            self.time_step,
            self.target_time,
        )

    def with_boundary_condition(
        self, boundary_condition: BoundaryCondition, /
    ) -> SolutionBuilder[
        PdeT,
        SolverT,
        InitialConditionT,
        SpatialStepT,
        BoundaryCondition,
        TimeStepT,
        TargetTimeT,
    ]:
        """
        Return a copy of this builder with the boundary condition replaced.

        Parameters
        ----------
        boundary_condition : BoundaryCondition
            The boundary condition of the PDE.

        Returns
        -------
        SolutionBuilder
            A new builder holding ``boundary_condition``.

        """
        return SolutionBuilder(
            self.pde,
            self.solver,
            self.initial_condition,
            self.spatial_step,
            boundary_condition,
            self.time_step,
            self.target_time,
        )

    def with_time_step(
        self, time_step: DType, /
    ) -> SolutionBuilder[
        PdeT,
        SolverT,
        InitialConditionT,
        SpatialStepT,
        BoundaryConditionT,
        DType,
        TargetTimeT,
    ]:
        """
        Return a copy of this builder with the time step replaced.

        Parameters
        ----------
        time_step : DType
            Time increment used by the solver's emulation.

        Returns
        -------
        SolutionBuilder
            A new builder holding ``time_step``.

        """
        return SolutionBuilder(
            self.pde,
            self.solver,
            self.initial_condition,
            self.spatial_step,
            self.boundary_condition,
            time_step,
            self.target_time,
        )

    def with_target_time(
        self, target_time: DType, /
    ) -> SolutionBuilder[
        PdeT,
        SolverT,
        InitialConditionT,
        SpatialStepT,
        BoundaryConditionT,
        TimeStepT,
        DType,
    ]:
        """
        Return a copy of this builder with the target time replaced.

        Parameters
        ----------
        target_time : DType
            Time at which the PDE's state is requested.

        Returns
        -------
        SolutionBuilder
            A new builder holding ``target_time``.

        """
        return SolutionBuilder(
            self.pde,
            self.solver,
            self.initial_condition,
            self.spatial_step,
            self.boundary_condition,
            self.time_step,
            target_time,
        )

    def compute[P: PDE](self: CompleteSolutionBuilder[P]) -> NDArray:
        """
        Run the solver on the collected inputs.

        Only available when every field holds a real value and the PDE is
        compatible with the solver (i.e. ``self`` matches
        :data:`Complete`); otherwise this method is a static type error.

        Returns
        -------
        NDArray
            The PDE state at ``target_time``.

        """
        return self.solver(
            self.pde,
            self.initial_condition,
            self.spatial_step,
            self.boundary_condition,
            self.time_step,
            self.target_time,
        )


type CompleteSolutionBuilder[P: PDE] = SolutionBuilder[
    P, Solver[P], NDArray, Vector, BoundaryCondition, DType, DType
]
"""A builder with all fields set and a PDE compatible with its solver."""

EMPTY_SOLUTION_BUILDER: SolutionBuilder = SolutionBuilder(
    PDE_UNSET,
    SOLVER_UNSET,
    INITIAL_CONDITION_UNSET,
    SPATIAL_STEP_UNSET,
    BOUNDARY_CONDITION_UNSET,
    TIME_STEP_UNSET,
    TARGET_TIME_UNSET,
)
"""Empty builder with every field unset; entry point for the ``with_*`` chain."""

__all__ = [
    "BOUNDARY_CONDITION_UNSET",
    "EMPTY_SOLUTION_BUILDER",
    "INITIAL_CONDITION_UNSET",
    "PDE_UNSET",
    "SOLVER_UNSET",
    "SPATIAL_STEP_UNSET",
    "TARGET_TIME_UNSET",
    "TIME_STEP_UNSET",
    "BoundaryConditionUnset",
    "CompleteSolutionBuilder",
    "InitialConditionUnset",
    "PdeUnset",
    "SolutionBuilder",
    "SolverUnset",
    "SpatialStepUnset",
    "TargetTimeUnset",
    "TimeStepUnset",
]
