from typing import Tuple

import torch

from heimdall.embeddings.position_embedder import PositionalEmbedder1D


class PatchEmbedder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        add_cls_token: bool = True,
        channel_last: bool = True,
    ):
        super(PatchEmbedder, self).__init__()

        self.channel_last = channel_last
        self.add_cls_token = add_cls_token

        self.patcher = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.position_embedder = PositionalEmbedder1D(d_model=out_channels)

        if self.add_cls_token:
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, out_channels))

    def forward(
        self, x: torch.Tensor, extract_pre_flat_shape: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim < 2:
            raise ValueError(
                f"Expected x to have atleast 2 dimensions, got {x.ndim} instead"
            )

        # Add a channel dimension: X[B, H, W] -> X[B, 1, H, W]
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.patcher(x)
        hw_shape = (x.shape[-2], x.shape[-1])

        # Flatten: X[B, C, H, W] -> X[B, C, h * w], where h << H, w << W
        x = x.flatten(2).transpose(1, 2)

        if self.add_cls_token:
            cls_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        pos_embeddings = self.position_embedder(x)
        pos_cls_embeddings = x + pos_embeddings

        if not self.channel_last:
            pos_cls_embeddings = pos_cls_embeddings.transpose(1, 2)

        if extract_pre_flat_shape:
            return pos_cls_embeddings, hw_shape
        else:
            return pos_cls_embeddings


class TemporalPatchEmbedder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 4,
        temporal_size: int = 1,
        add_cls_token: bool = True,
        channel_last: bool = True,
    ):
        super().__init__()

        self.channel_last = channel_last
        self.add_cls_token = add_cls_token

        self.patcher = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(temporal_size, patch_size, patch_size),
            stride=(temporal_size, patch_size, patch_size),
        )

        self.position_embedder = PositionalEmbedder1D(d_model=out_channels)

        if self.add_cls_token:
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, out_channels))

    def forward(
        self, x: torch.Tensor, extract_pre_flat_shape: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim < 4:
            raise ValueError(
                f"Expected x to have atleast 3 dimensions, got {x.ndim} instead"
            )

        # Add a temporal dimension: X[B, C, H, W] -> X[B, C, 1, H, W]
        if x.ndim == 4:
            x = x.unsqueeze(2)

        # Generate patch embeddings: X[B, C, T, H, W] -> X[B, C, t, h, w], where t << T, h << H, w << W
        x = self.patcher(x)
        thw_shape = (x.shape[-3], x.shape[-2], x.shape[-1])

        # Flatten: X[B, C, t, h, w] -> X[B, C, t * h * w]
        x = x.flatten(2).transpose(1, 2)

        if self.add_cls_token:
            cls_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        pos_embeddings = self.position_embedder(x)
        pos_cls_embeddings = x + pos_embeddings

        if not self.channel_last:
            pos_cls_embeddings = pos_cls_embeddings.transpose(1, 2)

        if extract_pre_flat_shape:
            return pos_cls_embeddings, thw_shape
        else:
            return pos_cls_embeddings
