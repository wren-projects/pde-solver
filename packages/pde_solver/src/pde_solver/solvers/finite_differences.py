from typing import final, override

import numpy as np

from pde_solver.abc.solver import Solver
from pde_solver.operators import divergence, gradient
from pde_solver.pde import HomogeneousVectorAdvectionMatrixDiffusionPDE
from pde_solver.pde_types import DType, NDArray, Vector


@final
class FiniteDifferences(Solver[HomogeneousVectorAdvectionMatrixDiffusionPDE]):
    """Solver using the Finite Differences method."""

    @override
    def _compute_time_step(
        self,
        pde: HomogeneousVectorAdvectionMatrixDiffusionPDE,
        state: NDArray,
        spacial_step: Vector,
        time_step: DType,
    ) -> NDArray:
        # the PDE is in the format:
        #   uₜ + ∇⋅(𝐚 ⋅ u) + ∇⋅(𝐁 ⋅ ∇u) = f = 0
        # which means that
        #   uₜ = -(∇⋅(𝐚 ⋅ u) + ∇⋅(𝐁 ⋅ ∇u))

        # α = ∇⋅(𝐚 ⋅ u)
        advection_flux = np.tensordot(pde.vector_advection, state, axes=0)
        advection_term = divergence(advection_flux, spacial_step)

        # β = ∇⋅(𝐁 ⋅ ∇u)
        state_gradient = gradient(state, spacial_step)
        diffusion_flux = np.tensordot(pde.matrix_diffusion, state_gradient, axes=1)
        diffusion_term = divergence(diffusion_flux, spacial_step)

        # uₜ = -(α + β) * Δt
        return -(advection_term + diffusion_term) * time_step
