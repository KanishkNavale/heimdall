import torch

from heimdall.transformers.multihead_attention import MultiHeadAttention


class LayerNorm(torch.nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dims))
        self.bias = torch.nn.Parameter(torch.zeros(dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, eps=self.eps
        )


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_scale: int = 4,
        dropout: float = 0.1,
    ):
        super(MLP, self).__init__()

        self.module = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim * hidden_dim_scale),
            torch.nn.GELU(),
            torch.nn.Linear(input_dim * hidden_dim_scale, input_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim: int, head_dim: int, n_heads: int) -> None:
        super(EncoderLayer, self).__init__()

        self.ln1 = LayerNorm(input_dim)
        self.ln2 = LayerNorm(input_dim)
        self.attn = MultiHeadAttention(input_dim, head_dim, n_heads)
        self.mlp = MLP(input_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(torch.nn.Module):
    def __init__(
        self, input_dim: int, head_dim: int, n_attention_head: int, n_layers: int
    ) -> None:
        super(Encoder, self).__init__()

        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(input_dim, head_dim, n_attention_head)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
