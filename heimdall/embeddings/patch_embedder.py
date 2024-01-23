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
    ):
        super(PatchEmbedder, self).__init__()

        self.flat_dispatch = flat_dispatch

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
        else:
            self.position_embedder = PositionalEmbedder2D(d_model=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.flat_dispatch:
            x = x.flatten(-2, -1)
            pos_embeddings = self.position_embedder(x.permute(0, 2, 1))
            return x + pos_embeddings.permute(0, 2, 1)
        else:
            pos_embeddings = self.position_embedder(x.permute(0, 2, 3, 1))
            return x + pos_embeddings.permute(0, 3, 1, 2)
