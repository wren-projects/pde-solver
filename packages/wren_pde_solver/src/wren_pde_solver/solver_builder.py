from __future__ import annotations

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


class SolverBuilder[T: PDE]:
    """
    A shorthand for running Solvers.

    Each instance represents one problem/situation which we need to solve. We can set
    all the necessary fields one by one, rather than having to provide them all at once.
    The object itself is not supposed to be changed, so all "set" methods return a new
    copy of the object instead.
    """

    @staticmethod
    def create() -> SolverBuilderInner[
        T,
        PdeNotSet,
        SolverNotSet,
        InitialConditionNotSet,
        SpatialStepNotSet,
        BoundaryConditionNotSet,
        TimeStepNotSet,
        TargetTimeNotSet,
    ]:
        """Create a SolverBuilderInner instance for a given PDE type."""
        return SolverBuilder.SolverBuilderInner()

    class SolverBuilderInner[
        S: PDE,
        S_PDE: PdeStatus,
        S_SOLVER: SolverStatus,
        S_INITIAL_CONDITION: InitialConditionStatus,
        S_SPATIAL_STEP: SpatialStepStatus,
        S_BOUNDARY_CONDITION: BoundaryConditionStatus,
        S_TIME_STEP: TimeStepStatus,
        S_TARGET_TIME: TargetTimeStatus,
    ]:
        """
        Represents one specific situation in which a PDE is to be computed.

        Should not be created directly. Instead, use the SolverBuilder.create() method.
        """

        # ruff: noqa: PLR0913
        def __init__(
            self,
            pde: S | None = None,
            solver: Solver[S] | None = None,
            initial_condition: NDArray | None = None,
            spatial_step: Vector | None = None,
            boundary_condition: BoundaryCondition | None = None,
            time_step: DType | None = None,
            target_time: DType | None = None,
        ) -> None:
            """
            Initialize a SolverBuilderInner instance.

            For private use only.
            """
            self.pde: S | None = pde
            self.solver: Solver[S] | None = solver
            self.initial_condition: NDArray | None = initial_condition
            self.spatial_step: Vector | None = spatial_step
            self.boundary_condition: BoundaryCondition | None = boundary_condition
            self.time_step: DType | None = time_step
            self.target_time: DType | None = target_time

        def set_pde(
            self, pde: S
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            PdeSet,
            S_SOLVER,
            S_INITIAL_CONDITION,
            S_SPATIAL_STEP,
            S_BOUNDARY_CONDITION,
            S_TIME_STEP,
            S_TARGET_TIME,
        ]:
            """
            Set the PDE which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
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
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            S_PDE,
            SolverSet,
            S_INITIAL_CONDITION,
            S_SPATIAL_STEP,
            S_BOUNDARY_CONDITION,
            S_TIME_STEP,
            S_TARGET_TIME,
        ]:
            """
            Set the solver which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
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
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            S_PDE,
            S_SOLVER,
            InitialConditionSet,
            S_SPATIAL_STEP,
            S_BOUNDARY_CONDITION,
            S_TIME_STEP,
            S_TARGET_TIME,
        ]:
            """
            Set the initial condition which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
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
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            S_PDE,
            S_SOLVER,
            S_INITIAL_CONDITION,
            SpatialStepSet,
            S_BOUNDARY_CONDITION,
            S_TIME_STEP,
            S_TARGET_TIME,
        ]:
            """
            Set the spatial step which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
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
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            S_PDE,
            S_SOLVER,
            S_INITIAL_CONDITION,
            S_SPATIAL_STEP,
            BoundaryConditionSet,
            S_TIME_STEP,
            S_TARGET_TIME,
        ]:
            """
            Set the boundary condition which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
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
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            S_PDE,
            S_SOLVER,
            S_INITIAL_CONDITION,
            S_SPATIAL_STEP,
            S_BOUNDARY_CONDITION,
            TimeStepSet,
            S_TARGET_TIME,
        ]:
            """
            Set the time step which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
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
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            S_PDE,
            S_SOLVER,
            S_INITIAL_CONDITION,
            S_SPATIAL_STEP,
            S_BOUNDARY_CONDITION,
            S_TIME_STEP,
            TargetTimeSet,
        ]:
            """
            Set the target time which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
                self.pde,
                self.solver,
                self.initial_condition,
                self.spatial_step,
                self.boundary_condition,
                self.time_step,
                target_time,
            )

        def compute(
            self: SolverBuilder.SolverBuilderInner[
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


type SolverBuilderReady[S: PDE] = SolverBuilder.SolverBuilderInner[
    S,
    PdeSet,
    SolverSet,
    InitialConditionSet,
    SpatialStepSet,
    BoundaryConditionSet,
    TimeStepSet,
    TargetTimeSet,
]


__all__ = ["SolverBuilder", "SolverBuilderReady"]
