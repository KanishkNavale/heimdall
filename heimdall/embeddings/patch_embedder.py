import torch


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return (
            features.flatten(2).transpose(1, 2)
            if self.flat_dispatch
            else features.permute(0, 2, 3, 1)
        )
