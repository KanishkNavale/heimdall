from typing import List, Optional

import numpy as np
import torch

from heimdall.logger import OverWatch
from heimdall.utils import convert_numpy_to_tensor

logger = OverWatch("inverse_kinematics")


@convert_numpy_to_tensor
def compute_stable_inverse_jacobian(
    jacobian: np.ndarray | torch.Tensor,
    C: float = np.inf,
    W: Optional[np.ndarray | torch.Tensor] = None,
    temperature: float = 1.0,
    logging: bool = False,
    **kwargs,
) -> torch.Tensor | np.ndarray:
    # Source: https://www.user.tu-berlin.de/mtoussai/teaching/13-Robotics/02-kinematics.pdf

    if W is None:
        W = torch.eye(jacobian.shape[0]) * temperature

    inverse_w = torch.linalg.inv(W)
    transpose_jacobian = torch.transpose(jacobian, 0, 1)
    inverted_c = torch.eye(jacobian.shape[1]) * (1.0 / C)

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


@convert_numpy_to_tensor
def compute_planar_trajectory(
    intial_joint_position: np.ndarray | torch.Tensor,
    target_tcp_position: np.ndarray | torch.Tensor,
    jacobian_function: callable,
    forward_kinematics_function: callable,
    interpolation: str = "linear",
    max_iterations: int = 100,
    logging: bool = False,
    **kwargs,
) -> List[np.ndarray] | List[torch.Tensor]:
    interpolation_methods = ["linear", "smmoth"]
    if interpolation not in interpolation_methods:
        raise ValueError(
            f"Avaialble interpolation methods are: {interpolation_methods}"
        )

    q_init = intial_joint_position
    y_target = target_tcp_position
    y_init = forward_kinematics_function(q_init)

    smooth_factor = 1.0

    joint_trajectory: List[np.ndarray] | List[torch.Tensor] = [q_init]

    for i in range(1, max_iterations + 1):
        q_current = joint_trajectory[-1]
        jacobian = jacobian_function(q_current)
        y_current = forward_kinematics_function(q_current)

        if interpolation == "linear":
            y_next = y_init + (y_target - y_init) * (i / max_iterations)

        elif interpolation == "smooth":
            smooth_factor = torch.linalg.norm(y_target - y_current)

        else:
            y_next = y_target

        jacobian_inverse = compute_stable_inverse_jacobian(jacobian)
        q_next = q_current + smooth_factor * jacobian_inverse @ (y_next - y_current)

        joint_trajectory.append(q_next)

        if torch.allclose(y_next, y_current, atol=1e-4):
            break

    if logging:
        y_current = forward_kinematics_function(joint_trajectory[-1])
        error = torch.linalg.norm(y_target - y_current)
        logger.info(f"Planar trajectory generation completed with error: {error:.4f}")

    if kwargs.get("numpy_found"):
        with torch.no_grad():
            return [joint.cpu().numpy() for joint in joint_trajectory]

    return joint_trajectory
