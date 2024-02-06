import torch

from heimdall.utils import convert_numpy_to_tensor


def _euclidean_distance_(
    pointcloud: torch.Tensor, sampled_index: int, candidate_indices: torch.Tensor
) -> torch.Tensor:
    return torch.linalg.norm(
        pointcloud[sampled_index] - pointcloud[candidate_indices], dim=-1, ord=2
    )


def _geodesic_distance_(
    pointcloud: torch.Tensor, sampled_index: int, candidate_indices: torch.Tensor
) -> torch.Tensor:
    raise NotImplementedError("Geodesic distance is not implemented yet.")


@convert_numpy_to_tensor
def furthest_point_sampling(
    pointcloud: torch.Tensor,
    n_samples: int = 1000,
    distance_method="geodesic",
    **kwargs,
) -> torch.Tensor:
    distance_methods = {
        "euclidean": _euclidean_distance_,
        "geodesic": _geodesic_distance_,
    }

    distance_metric = distance_methods.get(distance_method, None)

    if distance_metric is None:
        raise ValueError(
            f"Valid options for distance method: {list(distance_methods.keys())}"
        )

    N, _ = pointcloud.shape
    n_samples = min(n_samples, N)

    candidate_indices = torch.arange(N, device=pointcloud.device)
    sampled_indices = torch.zeros(n_samples, dtype=torch.long, device=pointcloud.device)

    distances = torch.ones(N, dtype=torch.float32, device=pointcloud.device) * torch.inf

    random_index = torch.randint(0, N, (1,))
    sampled_indices[0] = random_index

    candidate_indices = candidate_indices[candidate_indices != random_index]

    for i in range(1, n_samples):
        latest_candidate = sampled_indices[i - 1]

        neighbor_distances = distance_metric(
            pointcloud, latest_candidate, candidate_indices
        )

        distances[candidate_indices] = torch.min(
            distances[candidate_indices], neighbor_distances
        )

        selected = torch.argmax(distances[candidate_indices])
        sampled_indices[i] = candidate_indices[selected]

        candidate_indices = candidate_indices[candidate_indices != selected]

    return pointcloud[sampled_indices]
