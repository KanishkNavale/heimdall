from typing import Optional

import numpy as np
import torch

from heimdall.logger import OverWatch
from heimdall.utils import convert_numpy_to_tensor, timeit

logger = OverWatch("inverse_kinematics")


@convert_numpy_to_tensor
def compute_stable_inverse_jacobian(
    jacobian: np.ndarray | torch.Tensor,
    C: float = np.inf,
    W: Optional[np.ndarray | torch.Tensor] = None,
    temperature: float = 1.0,
    logging: bool = False,
    **kwargs,
) -> torch.Tensor:
    if W is None:
        W = torch.eye(jacobian.shape[0]) * temperature

    inverse_w = torch.linalg.inv(W)
    transpose_jacobian = torch.transpose(jacobian, 0, 1)
    inverted_c = torch.eye(jacobian.shape[1]) * (1.0 / C)

    with timeit(
        title="Computing stable inverse jacobian", logger=logger
    ) if logging else None:
        pseudo_inverse_jacobian = (
            inverse_w
            @ transpose_jacobian
            @ torch.linalg.inv(jacobian @ inverse_w @ transpose_jacobian + inverted_c)
        )

    if logging:
        determinant = torch.linalg.det(jacobian @ pseudo_inverse_jacobian)
        singularity = (
            True
            if torch.allclose(determinant, torch.zeros_like(determinant))
            else False
        )
        logger.info(
            f"Singularity: {singularity}"
        ) if not singularity else logger.warning(f"Singularity: {singularity}")

    if kwargs.get("numpy_found"):
        with torch.no_grad():
            return pseudo_inverse_jacobian.cpu().numpy()

    return pseudo_inverse_jacobian
