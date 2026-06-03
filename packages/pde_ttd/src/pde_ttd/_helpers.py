from collections.abc import Iterable, Reversible

import numpy as np

from pde_ttd.math import dot_product, qr_rows
from pde_ttd.types import Core


def reverse_cores[DType: np.floating](
    cores: Reversible[Core[DType]],
) -> Iterable[Core[DType]]:
    return (core.T for core in reversed(cores))


def orthogonalize_right[DType: np.floating](cores: list[Core[DType]]) -> None:
    for k in range(len(cores), 1, -1):  # for k = d to 2 step -1
        # [𝐆ₖ(βₖ₋₁; iₖβₖ), R(αₖ₋₁, βₖ₋₁)] := QR_rows(𝐆ₖ(αₖ₋₁; iₖβₖ))
        # G = 𝐆ₖ(αₖ₋₁; iₖβₖ)
        core = cores[k - 1]
        alpha_k1, i_k, beta_k = core.shape
        # 𝐐, 𝐑 = QR_rows(𝐆ₖ(αₖ₋₁; iₖβₖ)) = QR(𝐆ₖ(αₖ₋₁; iₖβₖ)ᵀ)ᵀ
        q, r = qr_rows(core.reshape((alpha_k1, i_k * beta_k)))
        # 𝐆ₖ(βₖ₋₁; iₖβₖ) = 𝐐
        cores[k - 1] = q.reshape((-1, i_k, beta_k))
        # 𝐆ₖ₋₁ := 𝐆ₖ₋₁ ×₃ 𝐑
        # NOTE: there is a typo in the TTD paper: it incorrectly says 𝐆ₖ ×₃ 𝐑
        cores[k - 2] = dot_product(cores[k - 2], r)
