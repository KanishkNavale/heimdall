import torch

from heimdall.embeddings.position_embedder import PositionalEmbedder2D


class PatchEmbedder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        add_cls_token: bool = True,
        flat_dispatch: bool = True,
        channel_last: bool = True,
    ):
        super(PatchEmbedder, self).__init__()

        self.flat_dispatch = flat_dispatch
        self.channel_last = channel_last
        self.add_cls_token = add_cls_token

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.position_embedder = PositionalEmbedder2D(d_model=out_channels)

        if self.add_cls_token:
            self.class_token = torch.nn.Parameter(torch.randn(1, 1, 1, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.add_cls_token:
            x = torch.cat([self.class_token, x], dim=2)

        pos_embeddings = self.position_embedder(x)
        pos_cls_embeddings = x + pos_embeddings

        if self.channel_last and self.flat_dispatch:
            return pos_cls_embeddings.permute(0, 2, 3, 1).flatten(1, 2)

        elif self.channel_last:
            return pos_cls_embeddings.permute(0, 2, 3, 1)

        elif self.flat_dispatch:
            return pos_cls_embeddings.flatten(2, 3)

        else:
            return pos_cls_embeddings
