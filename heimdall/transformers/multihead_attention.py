import math
from typing import Callable

import torch


class AttentionRegistry:
    def __init__(self) -> None:
        self.registry = {
            "scaled_dot_product": self.scaled_dot_product_attention,
            "fast_dot_product": self.fast_dot_product_attention,
            "skipped_dot_product": self.skipped_dot_product_attention,
        }

    @staticmethod
    def fast_dot_product_attention(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
        )

    @staticmethod
    def scaled_dot_product_attention(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        attention_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1))
        attention = torch.nn.functional.softmax(attention_logits, dim=-1)
        return torch.matmul(attention, V)

    @staticmethod
    def skipped_dot_product_attention(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        attention = AttentionRegistry.fast_dot_product_attention(Q, K, V)
        return attention + Q

    def get(self, attention_method: str) -> Callable:
        fetched_method = self.registry.get(attention_method, None)

        if fetched_method is None:
            raise ValueError(
                f"{attention_method} is not a valid attention method. Please choose from {list(self.registry.keys())}"
            )

        return fetched_method


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int = 512,
        n_head: int = 8,
        attention_method: str = "fast_dot_product",
    ):
        super(MultiHeadAttention, self).__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim
        self.n_head = n_head

        self.Q = torch.nn.Linear(input_dim, head_dim * n_head)
        self.K = torch.nn.Linear(input_dim, head_dim * n_head)
        self.V = torch.nn.Linear(input_dim, head_dim * n_head)

        self.dispatcher = torch.nn.Linear(head_dim * n_head, input_dim)

        attention_registry = AttentionRegistry()
        self.attention_method = attention_registry.get(attention_method)

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
