import torch

from heimdall.utils import convert_numpy_to_tensor


@convert_numpy_to_tensor
def furthest_point_sampling(
    pointcloud: torch.Tensor, n_samples: int = 1000, **kwargs
) -> torch.Tensor:
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

        neighbor_distances = torch.linalg.norm(
            pointcloud[latest_candidate] - pointcloud[candidate_indices], dim=-1
        )

        distances[candidate_indices] = torch.min(
            distances[candidate_indices], neighbor_distances
        )

        selected = torch.argmax(distances[candidate_indices])
        sampled_indices[i] = candidate_indices[selected]

        candidate_indices = candidate_indices[candidate_indices != selected]

    return pointcloud[sampled_indices]
