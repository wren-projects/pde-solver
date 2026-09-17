from __future__ import annotations

from dataclasses import dataclass
from typing import cast, final

from wren_pde_solver.abc.boundary import BoundaryCondition
from wren_pde_solver.abc.pde import PDE
from wren_pde_solver.abc.solver import Solver
from wren_pde_solver.pde_types import DType, NDArray, Vector


class PdeSet: ...


class PdeNotSet: ...


class SolverSet: ...


class SolverNotSet: ...


class InitialConditionSet: ...


class InitialConditionNotSet: ...


class SpatialStepSet: ...


class SpatialStepNotSet: ...


class BoundaryConditionSet: ...


class BoundaryConditionNotSet: ...


class TimeStepSet: ...


class TimeStepNotSet: ...


class TargetTimeSet: ...


class TargetTimeNotSet: ...


type PdeStatus = PdeSet | PdeNotSet
type SolverStatus = SolverSet | SolverNotSet
type InitialConditionStatus = InitialConditionSet | InitialConditionNotSet
type SpatialStepStatus = SpatialStepSet | SpatialStepNotSet
type BoundaryConditionStatus = BoundaryConditionSet | BoundaryConditionNotSet
type TimeStepStatus = TimeStepSet | TimeStepNotSet
type TargetTimeStatus = TargetTimeSet | TargetTimeNotSet


class SolutionBuilder[T: PDE]:
    """
    A shorthand for running Solvers.

    Each instance represents one problem/situation which we need to solve. We can set
    all the necessary fields one by one, rather than having to provide them all at once.
    The object itself is not supposed to be changed, so all "set" methods return a new
    copy of the object instead.
    """

    @staticmethod
    def create() -> SolutionBuilderInner[
        T,
        PdeNotSet,
        SolverNotSet,
        InitialConditionNotSet,
        SpatialStepNotSet,
        BoundaryConditionNotSet,
        TimeStepNotSet,
        TargetTimeNotSet,
    ]:
        """Create a SolutionBuilderInner instance for a given PDE type."""
        return SolutionBuilder.SolutionBuilderInner()

    @final
    @dataclass(frozen=True)
    class SolutionBuilderInner[
        S: PDE,
        PdeS: PdeStatus,
        SolverS: SolverStatus,
        InitialConditionS: InitialConditionStatus,
        SpatialStepS: SpatialStepStatus,
        BoundaryConditionS: BoundaryConditionStatus,
        TimeStepS: TimeStepStatus,
        TargetTimeS: TargetTimeStatus,
    ]:
        """
        Represents one specific situation in which a PDE is to be computed.

        Should not be created directly. Use the SolutionBuilder.create() method instead.
        """

        pde: S | None = None
        solver: Solver[S] | None = None
        initial_condition: NDArray | None = None
        spatial_step: Vector | None = None
        boundary_condition: BoundaryCondition | None = None
        time_step: DType | None = None
        target_time: DType | None = None

        def get_pde(
            self: SolutionBuilder.SolutionBuilderInner[
                S,
                PdeSet,
                SolverS,
                InitialConditionS,
                SpatialStepS,
                BoundaryConditionS,
                TimeStepS,
                TargetTimeS,
            ],
        ) -> S:
            """
            Return the set pde.

            Can be called only if the pde was already set.
            """
            return cast(S, self.pde)

        def get_solver(
            self: SolutionBuilder.SolutionBuilderInner[
                S,
                PdeS,
                SolverSet,
                InitialConditionS,
                SpatialStepS,
                BoundaryConditionS,
                TimeStepS,
                TargetTimeS,
            ],
        ) -> Solver[S]:
            """
            Return the set solver.

            Can be called only if the solver was already set.
            """
            return cast(Solver[S], self.solver)

        def get_initial_condition(
            self: SolutionBuilder.SolutionBuilderInner[
                S,
                PdeS,
                SolverS,
                InitialConditionSet,
                SpatialStepS,
                BoundaryConditionS,
                TimeStepS,
                TargetTimeS,
            ],
        ) -> NDArray:
            """
            Return the set initial condition.

            Can be called only if the initial condition was already set.
            """
            return cast(NDArray, self.initial_condition)

        def get_spatial_step(
            self: SolutionBuilder.SolutionBuilderInner[
                S,
                PdeS,
                SolverS,
                InitialConditionS,
                SpatialStepSet,
                BoundaryConditionS,
                TimeStepS,
                TargetTimeS,
            ],
        ) -> Vector:
            """
            Return the set spacial step.

            Can be called only if the spacial step was already set.
            """
            return cast(Vector, self.spatial_step)

        def get_boundary_condition(
            self: SolutionBuilder.SolutionBuilderInner[
                S,
                PdeS,
                SolverS,
                InitialConditionS,
                SpatialStepS,
                BoundaryConditionSet,
                TimeStepS,
                TargetTimeS,
            ],
        ) -> BoundaryCondition:
            """
            Return the set boundary condition.

            Can be called only if the boundary condition already set.
            """
            return cast(BoundaryCondition, self.boundary_condition)

        def get_time_step(
            self: SolutionBuilder.SolutionBuilderInner[
                S,
                PdeS,
                SolverS,
                InitialConditionS,
                SpatialStepS,
                BoundaryConditionS,
                TimeStepSet,
                TargetTimeS,
            ],
        ) -> DType:
            """
            Return the set time step.

            Can be called only if the time step was already set.
            """
            return cast(DType, self.time_step)

        def get_target_time(
            self: SolutionBuilder.SolutionBuilderInner[
                S,
                PdeS,
                SolverS,
                InitialConditionS,
                SpatialStepS,
                BoundaryConditionS,
                TimeStepS,
                TargetTimeSet,
            ],
        ) -> DType:
            """
            Return the set target time.

            Can be called only if the target time was already set.
            """
            return cast(DType, self.target_time)

        def set_pde(
            self, pde: S
        ) -> SolutionBuilder.SolutionBuilderInner[
            S,
            PdeSet,
            SolverS,
            InitialConditionS,
            SpatialStepS,
            BoundaryConditionS,
            TimeStepS,
            TargetTimeS,
        ]:
            """
            Set the PDE which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolutionBuilder.SolutionBuilderInner(
                pde,
                self.solver,
                self.initial_condition,
                self.spatial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_solver(
            self, solver: Solver[S]
        ) -> SolutionBuilder.SolutionBuilderInner[
            S,
            PdeS,
            SolverSet,
            InitialConditionS,
            SpatialStepS,
            BoundaryConditionS,
            TimeStepS,
            TargetTimeS,
        ]:
            """
            Set the solver which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolutionBuilder.SolutionBuilderInner(
                self.pde,
                solver,
                self.initial_condition,
                self.spatial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_initial_condition(
            self, initial_condition: NDArray
        ) -> SolutionBuilder.SolutionBuilderInner[
            S,
            PdeS,
            SolverS,
            InitialConditionSet,
            SpatialStepS,
            BoundaryConditionS,
            TimeStepS,
            TargetTimeS,
        ]:
            """
            Set the initial condition which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolutionBuilder.SolutionBuilderInner(
                self.pde,
                self.solver,
                initial_condition,
                self.spatial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_spatial_step(
            self, spatial_step: Vector
        ) -> SolutionBuilder.SolutionBuilderInner[
            S,
            PdeS,
            SolverS,
            InitialConditionS,
            SpatialStepSet,
            BoundaryConditionS,
            TimeStepS,
            TargetTimeS,
        ]:
            """
            Set the spatial step which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolutionBuilder.SolutionBuilderInner(
                self.pde,
                self.solver,
                self.initial_condition,
                spatial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_boundary_condition(
            self, boundary_condition: BoundaryCondition
        ) -> SolutionBuilder.SolutionBuilderInner[
            S,
            PdeS,
            SolverS,
            InitialConditionS,
            SpatialStepS,
            BoundaryConditionSet,
            TimeStepS,
            TargetTimeS,
        ]:
            """
            Set the boundary condition which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolutionBuilder.SolutionBuilderInner(
                self.pde,
                self.solver,
                self.initial_condition,
                self.spatial_step,
                boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_time_step(
            self, time_step: DType
        ) -> SolutionBuilder.SolutionBuilderInner[
            S,
            PdeS,
            SolverS,
            InitialConditionS,
            SpatialStepS,
            BoundaryConditionS,
            TimeStepSet,
            TargetTimeS,
        ]:
            """
            Set the time step which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolutionBuilder.SolutionBuilderInner(
                self.pde,
                self.solver,
                self.initial_condition,
                self.spatial_step,
                self.boundary_condition,
                time_step,
                self.target_time,
            )

        def set_target_time(
            self, target_time: DType
        ) -> SolutionBuilder.SolutionBuilderInner[
            S,
            PdeS,
            SolverS,
            InitialConditionS,
            SpatialStepS,
            BoundaryConditionS,
            TimeStepS,
            TargetTimeSet,
        ]:
            """
            Set the target time which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolutionBuilder.SolutionBuilderInner(
                self.pde,
                self.solver,
                self.initial_condition,
                self.spatial_step,
                self.boundary_condition,
                self.time_step,
                target_time,
            )

        def compute(
            self: SolutionBuilder.SolutionBuilderInner[
                S,
                PdeSet,
                SolverSet,
                InitialConditionSet,
                SpatialStepSet,
                BoundaryConditionSet,
                TimeStepSet,
                TargetTimeSet,
            ],
        ) -> NDArray:
            """
            Compute the state at the given time of the given situation.

            A shorthand for Solver.__call__.

            Here, a partial differential equation is a triple of the PDE itself, the
            initial condition, and the boundary condition.
            """
            if (
                self.solver is None
                or self.pde is None
                or self.initial_condition is None
                or self.spatial_step is None
                or self.boundary_condition is None
                or self.time_step is None
                or self.target_time is None
            ):
                raise ValueError("Cannot compute an instance with some values None")
            return self.solver(
                self.pde,
                self.initial_condition,
                self.spatial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )


type SolutionBuilderReady[S: PDE] = SolutionBuilder.SolutionBuilderInner[
    S,
    PdeSet,
    SolverSet,
    InitialConditionSet,
    SpatialStepSet,
    BoundaryConditionSet,
    TimeStepSet,
    TargetTimeSet,
]


__all__ = ["SolutionBuilder", "SolutionBuilderReady"]
