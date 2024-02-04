import torch

from heimdall.datatypes.pose import SO3, Pose


def relative_pose(source_pose: Pose, target_pose: Pose) -> Pose:
    return Pose.from_SE3(target_pose.SE3 @ source_pose.inverted.SE3)


def kabsch_transform(source: torch.Tensor, target: torch.Tensor) -> Pose:
    if source.shape[-1] != 3 or target.shape[-1] != 3 or source.ndim != target.ndim:
        raise ValueError("Source and target must have shape (N, 3) or (B, N, 3)")

    if source.ndim == 2 and target.ndim == 2:
        batched_source = source[None, ...]
        batched_target = target[None, ...]
    else:
        batched_source = source
        batched_target = target

    centered_source = batched_source - batched_source.mean(dim=-2, keepdim=True)
    centered_target = batched_target - batched_target.mean(dim=-2, keepdim=True)

    covariance_matrix = centered_source.transpose(-2, -1) @ centered_target
    U, _, VT = torch.linalg.svd(covariance_matrix)
    V = VT.transpose(-2, -1)
    UT = U.transpose(-2, -1)

    sign = torch.sign(torch.linalg.det(V @ UT))
    V[:, -1] *= sign[..., None]

    rotation_matrix = V @ UT
    translation = batched_target.mean(dim=-2) - batched_source.mean(dim=-2)

    if source.ndim == 2 and target.ndim == 2:
        return Pose(translation=translation[0], SO3=SO3(rotation_matrix[0]))

    return Pose(translation=translation, SO3=SO3(rotation_matrix))
