import torch

from heimdall.datatypes.pose import SO3, Pose


def relative_pose(source_pose: Pose, target_pose: Pose) -> Pose:
    return Pose.from_SE3(target_pose.SE3 @ source_pose.inverted.SE3)


def kabsch_transform(source: torch.Tensor, target: torch.Tensor) -> Pose:
    if source.ndim != target.ndim:
        raise ValueError(
            "Source and target must have same number of dimensions. Preferably N x 3 or B x N x 3"
        )

    if source.ndim == 2 and target.ndim == 2:
        source = source[None, ...]
        target = target[None, ...]

    centered_source = source - source.mean(dim=-2)
    centered_target = target - target.mean(dim=-2)

    covariance_matrix = centered_source.transpose(-2, -1) @ centered_target
    U, _, VT = torch.linalg.svd(covariance_matrix)
    V = VT.transpose(-2, -1)
    UT = U.transpose(-2, -1)

    sign = torch.sign(torch.linalg.det(V @ UT))
    V[:, -1] *= sign[..., None]

    rotation_matrix = V @ UT
    translation = target.mean(dim=-2) - source.mean(dim=-2)

    return Pose(translation=translation.squeeze(-1), SO3=SO3(rotation_matrix))
