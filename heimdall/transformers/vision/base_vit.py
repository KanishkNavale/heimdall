import torch

from heimdall.embeddings.patch_embedder import PatchEmbedder
from heimdall.transformers.encoder_layer import Encoder


class BaseViTFeature(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        encoder_head_size: int,
        n_attention_heads: int,
        n_encoder_layers: int,
    ) -> None:
        super(BaseViTFeature, self).__init__()

        self.patch_embedder = PatchEmbedder(in_channels, out_channels, patch_size)
        self.encoder = Encoder(
            out_channels, encoder_head_size, n_attention_heads, n_encoder_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder.forward(x, extract_pre_flat_shape=False)
        return self.encoder(x)

    def compute_classification_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 0, :]
