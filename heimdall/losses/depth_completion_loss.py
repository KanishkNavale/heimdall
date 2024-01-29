import torch
from kornia.geometry.depth import warp_frame_depth


class DepthCompletionLoss:
    def __init__(self, yaml_config_path: str) -> None:
        self.name = "sereact depth completion loss"

        # Init. configuration
        # config_dictionary = initialize_config_file(yaml_config_path)
        # self.config = DepthCompletionMLPConfig.from_dictionary(config_dictionary)

        # deconstruct
        self.reduction = self.config.loss.reduction
        self.invert_depth = self.config.loss.invert_depth

    def _reducer(self, batchwise_loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return torch.mean(batchwise_loss)
        else:
            return torch.mean(batchwise_loss)

    @staticmethod
    def _masked_depth_loss(
        groundtruth_depth: torch.Tensor, predicted_depth: torch.Tensor
    ) -> torch.Tensor:
        non_zero_depth_mask = torch.where(groundtruth_depth > 0.0, 1.0, 0.0)
        difference = predicted_depth - groundtruth_depth
        masked_differences = non_zero_depth_mask * difference

        distances = torch.linalg.norm(
            masked_differences.squeeze(dim=1), ord=2, dim=(-2, -1)
        )

        return torch.square(distances)

    def _depth_smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :-2] - x[:, :, 1:-1, 2:]
        vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:-1] - x[:, :, 2:, 1:-1]
        der_2nd = horizontal.abs() + vertical.abs()
        return torch.norm(der_2nd, p=1, dim=(-3, -2, -1))

    @staticmethod
    def _reprojection_loss(
        source_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        predicted_target_depth: torch.Tensor,
        groundtruth_traget_depth: torch.Tensor,
        source_to_target_trafo: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> torch.Tensor:
        warped_image = warp_frame_depth(
            source_rgb,
            predicted_target_depth,
            source_to_target_trafo.squeeze(dim=1),
            intrinsic.squeeze(dim=1),
        )

        neg_zero_mask = torch.where(groundtruth_traget_depth <= 0.0, 1.0, 0.0)
        scaling_factors = torch.sum(neg_zero_mask, dim=(-2, -1)).squeeze(dim=-1)

        masked_difference = neg_zero_mask * (warped_image - target_rgb)
        distances = torch.norm(masked_difference, p=1, dim=(-3, -2, -1))

        return distances / torch.max(scaling_factors, torch.ones_like(scaling_factors))

    def __call__(
        self,
        rgb_a: torch.Tensor,
        rgb_b: torch.Tensor,
        trafo: torch.Tensor,
        depth_b: torch.Tensor,
        predicted_depth_b: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        masked_depth_loss = self._masked_depth_loss(depth_b, predicted_depth_b)
        smoothness_depth_loss = self._depth_smoothness_loss(predicted_depth_b)
        reprojection_loss = self._reprojection_loss(
            rgb_a, rgb_b, predicted_depth_b, depth_b, trafo, intrinsics
        )

        reduced_losses = torch.vstack(
            [
                self._reducer(masked_depth_loss),
                self._reducer(smoothness_depth_loss),
                self._reducer(reprojection_loss),
            ]
        )

        weights = torch.as_tensor(
            [1.0, 1e-1, 1e-1], device=reduced_losses.device, dtype=reduced_losses.dtype
        )

        return weights @ reduced_losses
