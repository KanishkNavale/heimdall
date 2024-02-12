from typing import List

import torch

from heimdall.embeddings.patch_embedder import PatchEmbedder, TemporalPatchEmbedder
from heimdall.transformers.encoder import Encoder
from heimdall.transformers.vision.encoder import MultiScaleEncoder


class ViT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        n_attention_heads: int,
        n_encoder_layers: int,
    ) -> None:
        super().__init__()

        self.patch_embedder = PatchEmbedder(in_channels, out_channels, patch_size)
        self.encoder = Encoder(
            out_channels, out_channels, n_attention_heads, n_encoder_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder.forward(x, extract_pre_flat_shape=False)
        return self.encoder(x)

    def compute_classification_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 0, :]


class MViT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        temporal_size: int,
        n_heads: int,
        block_config: List[int],
        add_cls_token: bool = True,
    ) -> None:
        super().__init__()

        self.temporal_patch_embedder = TemporalPatchEmbedder(
            in_channels,
            out_channels,
            patch_size,
            temporal_size,
            add_cls_token,
        )

        self.scale_blocks = torch.nn.ModuleDict({})
        input_dim = out_channels
        output_dim = input_dim * n_heads

        for i, block in enumerate(block_config):
            block_name = f"scale_block_{i}"
            self.scale_blocks[block_name] = MultiScaleEncoder(
                input_dim, output_dim, n_heads, block
            )

            input_dim = output_dim
            output_dim *= 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, thw_shape = self.temporal_patch_embedder.forward(
            x, extract_pre_flat_shape=True
        )

        for block in self.scale_blocks.values():
            x, thw_shape = block(x, thw_shape)
            pass

        return x

    def compute_classification_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 0, :]
