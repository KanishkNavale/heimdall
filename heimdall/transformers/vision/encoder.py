from typing import Tuple

import torch

from heimdall.transformers.encoder import MLP, LayerNorm
from heimdall.transformers.multi_head_pooled_attention import MultiHeadPooledAttention


class MultiScaleEncoderLayer(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, n_heads: int) -> None:
        super().__init__()

        self.ln1 = LayerNorm(input_dim)
        self.ln2 = LayerNorm(input_dim)
        self.attn = MultiHeadPooledAttention(input_dim, head_dim, n_heads)
        self.mlp = MLP(input_dim)

    def forward(self, x: torch.Tensor, thw_shape: Tuple[int, int, int]) -> torch.Tensor:
        x, thw_shape = self.attn(self.ln1(x), thw_shape)
        x += self.mlp(self.ln2(x))
        return x, thw_shape


class MultiScaleEncoder(torch.nn.Module):
    def __init__(
        self, input_dim: int, head_dim: int, n_heads: int, n_layers: int
    ) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                MultiScaleEncoderLayer(input_dim, head_dim, n_heads)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, thw_shape: Tuple[int, int, int]) -> torch.Tensor:
        for layer in self.layers:
            x, thw_shape = layer(x, thw_shape)
        return x, thw_shape
