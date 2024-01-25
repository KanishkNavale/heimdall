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
        x = self.patcher(x)
        hw_shape = (x.shape[-2], x.shape[-1])

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
