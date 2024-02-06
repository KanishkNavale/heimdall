import torch


def furthest_point_sampling(
    pointcloud: torch.Tensor, n_samples: int = 1000
) -> torch.Tensor:
    N, _ = pointcloud.shape
    n_samples = min(n_samples, N)

    sampled_indices = torch.zeros(n_samples, dtype=torch.long, device=pointcloud.device)

    distances = torch.ones(N, dtype=torch.float32, device=pointcloud.device) * torch.inf

    random_index = torch.randint(0, N, (1,))
    sampled_indices[0] = random_index

    for i in range(1, n_samples):
        last_added = sampled_indices[i - 1]

        distances = torch.min(
            distances, torch.norm(pointcloud - pointcloud[last_added], dim=1)
        )

        selected = torch.argmax(distances)
        sampled_indices[i] = selected

    return pointcloud[sampled_indices]
