from typing import Optional, Tuple

import torch


class AttentionPooler(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        has_cls_token: bool = True,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.pooler = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.has_cls_token = has_cls_token

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        # Check shapes
        H, W = hw_shape
        B, N, _, C = x.shape

        if self.has_cls_token:
            cls_tokens = x[:, :, 0, :].unsqueeze(2)
            x = x[:, :, 1:, :]

        # Reshape: X[B, N, L, C] -> X[B * N, C, H, W]
        x = x.transpose(1, 2).reshape(B * N, C, H, W)

        # Pooler: X[B * N, C, H, W] -> X[B * N, D, h, w], where h << H, w << W, C <= D
        pooled_features = self.pooler.forward(x)

        # Reshape: X[B * N, D, h, w] -> X[B, N, D, h, w]
        reshaped_pooled_features = pooled_features.reshape(
            B,
            N,
            pooled_features.shape[1],
            pooled_features.shape[2],
            pooled_features.shape[3],
        )

        # Flatten: X[B, N, D, h, w] -> X[B, N, h * w, D]
        flattened_pooled_features = reshaped_pooled_features.flatten(-2).transpose(
            -2, -1
        )

        # Add cls token: X[B, N, h * w, D] -> X[B, N, 1 + (h * w), D]
        if self.has_cls_token:
            flattened_pooled_features = torch.cat(
                [cls_tokens, flattened_pooled_features], dim=2
            )

        return flattened_pooled_features
