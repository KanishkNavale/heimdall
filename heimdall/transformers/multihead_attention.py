import math

import torch

# TODO:
# 1. Add support for different attention methods:  Additive Attention, etc.
# 2. Add support for dropout mechanisms: Head, Attention, etc.


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        n_head: int,
        attention_scale: float = 1.0,
        attention_method: str = "scaled_dot_product",
    ):
        super(MultiHeadAttention, self).__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim
        self.n_head = n_head

        self.Q = torch.nn.Linear(input_dim, head_dim * n_head)
        self.K = torch.nn.Linear(input_dim, head_dim * n_head)
        self.V = torch.nn.Linear(input_dim, head_dim * n_head)

        self.dispatcher = torch.nn.Linear(head_dim * n_head, input_dim)

        self.attention_scale = attention_scale
        attention_directory = {
            "scaled_dot_product": self.scaled_dot_product_attention,
        }

        self.attention_method = attention_directory.get(attention_method, None)
        if self.attention_method is None:
            raise ValueError(
                f"{attention_method} is not a valid attention method. Please choose from {list(attention_directory.keys())}"
            )

        # Initialize heads
        self._init_weights()
        self._init_bias()

    def _init_weights(self) -> None:
        torch.nn.init.xavier_uniform_(self.Q.weight)
        torch.nn.init.xavier_uniform_(self.K.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)

    def _init_bias(self) -> None:
        torch.nn.init.zeros_(self.Q.bias)
        torch.nn.init.zeros_(self.K.bias)
        torch.nn.init.zeros_(self.V.bias)

    def scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        attention_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(
            Q.size(-1) ** self.attention_scale
        )
        return torch.matmul(attention_weights, V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.size()

        # X[B: Batch Size, L: Sequence Length, D: Embedding Dim.] -> Q|K|V[B, L, N: No. of Heads * H:]
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # Q|K|V[B, L, N * H] -> Q|K|V[B, N, L, H]
        Q = Q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        # Compute Attention: Q|K|V[B, N, L, H] -> Attention[B, N, L, H]
        attention = self.attention_method(Q, K, V)

        # Concatenate heads: [B, N, L, H] -> [B, L, N * H]
        stacked_attention = attention.transpose(1, 2).contiguous().view(B, L, -1)

        # Project Attention: [B, L, N * H] -> [B, L, D]
        return self.dispatcher(stacked_attention)
