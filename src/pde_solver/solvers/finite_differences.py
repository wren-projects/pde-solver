from typing import final, override

import numpy as np

from pde_solver.abc.solver import Solver
from pde_solver.operators import divergence, gradient
from pde_solver.pde import HomogeneousVectorAdvectionMatrixDiffusionPDE
from pde_solver.pde_types import DType, NDArray, Vector


@final
class FiniteDifferences(Solver[HomogeneousVectorAdvectionMatrixDiffusionPDE]):
    """Solver using the Finite Differences method."""

    STABILITY_TRESHOLD = 0.5

    @override
    def _get_time_step(
        self,
        pde: HomogeneousVectorAdvectionMatrixDiffusionPDE,
        state: NDArray,
        spacial_discretization_step: Vector,
        time_discretization_step: DType,
    ) -> NDArray:
        # equation format is
        # 󰌵uₜ + ∇⋅(𝐚⋅u) + ∇⋅(𝐁⋅∇u) = f = 0
        u_time = -(
            divergence(
                np.tensordot(state, pde.vector_advection, axes=0),
                spacial_discretization_step,
            )
            + divergence(
                np.tensordot(
                    pde.matrix_diffusion,
                    gradient(state, spacial_discretization_step),
                    axes=(-1, 1),
                ),
                spacial_discretization_step,
            )
        )

        return u_time * time_discretization_step
