import torch

from heimdall.datatypes.pose import SO3, Pose


def rotational_geodesic_distance(source: SO3, target: SO3) -> torch.Tensor:
    if source.matrix.ndim != target.matrix.ndim:
        raise ValueError(
            "Source and target SO3 matrices must have the same number of dimensions."
        )

    relative_rotation = source.matrix @ target.matrix.transpose(-2, -1)

    trace = torch.sum(torch.diagonal(relative_rotation, dim1=-2, dim2=-1), dim=-1)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clip(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)


def pose_geodesic_distance(source: Pose, target: Pose) -> torch.Tensor:
    if source.SE3.ndim != target.SE3.ndim:
        raise ValueError(
            "Source and target SE3 matrices must have the same number of dimensions."
        )

    rotation_distance = rotational_geodesic_distance(source.SO3, target.SO3)
    translation_distance = torch.linalg.norm(
        target.translation - source.translation, dim=-1
    )
    return torch.sqrt(rotation_distance**2 + translation_distance**2)
