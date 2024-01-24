import torch

from heimdall.embeddings.position_embedder import (
    PositionalEmbedder1D,
    PositionalEmbedder2D,
)


class PatchEmbedder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        flat_dispatch: bool = False,
        channel_last: bool = True,
    ):
        super(PatchEmbedder, self).__init__()

        self.flat_dispatch = flat_dispatch
        self.channel_last = channel_last

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if self.flat_dispatch:
            self.position_embedder = PositionalEmbedder1D(
                d_model=out_channels, max_len=2000
            )
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, out_channels))
        else:
            self.position_embedder = PositionalEmbedder2D(d_model=out_channels)
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, 1, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.flat_dispatch:
            flat_x = x.flatten(-2, -1).permute(0, 2, 1)
            concat_cls_embeddings = torch.cat([self.class_token, flat_x], dim=1)
            pos_embeddings = self.position_embedder(concat_cls_embeddings)
            pos_cls_embeddings = concat_cls_embeddings + pos_embeddings
            return (
                pos_cls_embeddings
                if self.channel_last
                else pos_cls_embeddings.permute(0, 2, 1)
            )
        else:
            concat_cls_embeddings = torch.cat([self.class_token, x], dim=2)
            pos_embeddings = self.position_embedder(x)
            pos_cls_embeddings = concat_cls_embeddings + pos_embeddings
            return (
                pos_cls_embeddings
                if self.channel_last
                else pos_cls_embeddings.permute(0, 3, 1, 2)
            )
