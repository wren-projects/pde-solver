from __future__ import annotations

from typing import Literal

from wren_pde_solver.abc.boundary import BoundaryCondition
from wren_pde_solver.abc.pde import PDE
from wren_pde_solver.abc.solver import Solver
from wren_pde_solver.pde_types import DType, NDArray, Vector

type Yes = Literal[True]
type No = Literal[False]


class SolverBuilder[T: PDE]:
    """
    A shorthand for running Solvers.

    Each instance represents one problem/situation which we need to solve. We can set
    all the necessary fields one by one, rather than having to provide them all at once.
    The object itself is not supposed to be changed so all "set" methods return a new
    copy of the object instead.
    """

    @staticmethod
    def create() -> SolverBuilderInner[T, No, No, No, No, No, No, No]:
        """Create a SolverBuilderInner instance for a given PDE type."""
        return SolverBuilder.SolverBuilderInner()

    class SolverBuilderInner[
        S: PDE,
        PDE_SET: bool,
        SOLVER_SET: bool,
        INITIAL_CONDITION_SET: bool,
        SPACIAL_STEP_SET: bool,
        BOUNDARY_CONDITION_SET: bool,
        TIME_STEP_SET: bool,
        TARGET_TIME_SET: bool,
    ]:
        """
        Represents one specific situation in which a PDE is to be computed.

        Should not be create directly. Instead, use SolverBuilder.create() method.
        """

        # ruff: noqa: PLR0913
        def __init__(
            self,
            pde: S | None = None,
            solver: Solver[S] | None = None,
            initial_condition: NDArray | None = None,
            spacial_step: Vector | None = None,
            boundary_condition: BoundaryCondition | None = None,
            time_step: DType | None = None,
            target_time: DType | None = None,
        ) -> None:
            """
            Create a SolverBuilderInner method.

            For private use only.
            """
            self.pde: S | None = pde
            self.solver: Solver[S] | None = solver
            self.initial_condition: NDArray | None = initial_condition
            self.spacial_step: Vector | None = spacial_step
            self.boundary_condition: BoundaryCondition | None = boundary_condition
            self.time_step: DType | None = time_step
            self.target_time: DType | None = target_time

        def set_pde(
            self, pde: S
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            Yes,
            SOLVER_SET,
            INITIAL_CONDITION_SET,
            SPACIAL_STEP_SET,
            BOUNDARY_CONDITION_SET,
            TIME_STEP_SET,
            TARGET_TIME_SET,
        ]:
            """
            Set the solver which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
                pde,
                self.solver,
                self.initial_condition,
                self.spacial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_solver(
            self, solver: Solver[S]
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            PDE_SET,
            Yes,
            INITIAL_CONDITION_SET,
            SPACIAL_STEP_SET,
            BOUNDARY_CONDITION_SET,
            TIME_STEP_SET,
            TARGET_TIME_SET,
        ]:
            """
            Set the solver which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
                self.pde,
                solver,
                self.initial_condition,
                self.spacial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_initial_condition(
            self, initial_condition: NDArray
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            PDE_SET,
            SOLVER_SET,
            Yes,
            SPACIAL_STEP_SET,
            BOUNDARY_CONDITION_SET,
            TIME_STEP_SET,
            TARGET_TIME_SET,
        ]:
            """
            Set the initial condition which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
                self.pde,
                self.solver,
                initial_condition,
                self.spacial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_spacial_step(
            self, spacial_step: Vector
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            PDE_SET,
            SOLVER_SET,
            INITIAL_CONDITION_SET,
            Yes,
            BOUNDARY_CONDITION_SET,
            TIME_STEP_SET,
            TARGET_TIME_SET,
        ]:
            """
            Set the spacial step which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
                self.pde,
                self.solver,
                self.initial_condition,
                spacial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_boundary_condition(
            self, boundary_condition: BoundaryCondition
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            PDE_SET,
            SOLVER_SET,
            INITIAL_CONDITION_SET,
            SPACIAL_STEP_SET,
            Yes,
            TIME_STEP_SET,
            TARGET_TIME_SET,
        ]:
            """
            Set the boundary condition which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
                self.pde,
                self.solver,
                self.initial_condition,
                self.spacial_step,
                boundary_condition,
                self.time_step,
                self.target_time,
            )

        def set_time_step(
            self, time_step: DType
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            PDE_SET,
            SOLVER_SET,
            INITIAL_CONDITION_SET,
            SPACIAL_STEP_SET,
            BOUNDARY_CONDITION_SET,
            Yes,
            TARGET_TIME_SET,
        ]:
            """
            Set the time step which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
                self.pde,
                self.solver,
                self.initial_condition,
                self.spacial_step,
                self.boundary_condition,
                time_step,
                self.target_time,
            )

        def set_target_time(
            self, target_time: DType
        ) -> SolverBuilder.SolverBuilderInner[
            S,
            PDE_SET,
            SOLVER_SET,
            INITIAL_CONDITION_SET,
            SPACIAL_STEP_SET,
            BOUNDARY_CONDITION_SET,
            TIME_STEP_SET,
            TARGET_TIME_SET,
        ]:
            """
            Set the the target time which will be used for this situation.

            After all fields have been set, the "compute" method becomes available.
            """
            return SolverBuilder.SolverBuilderInner(
                self.pde,
                self.solver,
                self.initial_condition,
                self.spacial_step,
                self.boundary_condition,
                self.time_step,
                target_time,
            )

        def compute(
            self: SolverBuilder.SolverBuilderInner[
                T, Yes, Yes, Yes, Yes, Yes, Yes, Yes
            ],
        ) -> NDArray:
            """
            Compute the state at the given time of the given situaion.

            A shorthand for Solver.__call__.

            Here, a partial differential equation is a triple of the PDE itself, the
            initial condition, and the boundary condition.
            """
            if (
                self.solver is None
                or self.pde is None
                or self.initial_condition is None
                or self.spacial_step is None
                or self.boundary_condition is None
                or self.time_step is None
                or self.target_time is None
            ):
                raise ValueError("Cannot compute an instance with some values None")
            return self.solver(
                self.pde,
                self.initial_condition,
                self.spacial_step,
                self.boundary_condition,
                self.time_step,
                self.target_time,
            )


__all__ = ["SolverBuilder"]
