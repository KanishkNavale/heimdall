import math
from typing import Callable

import torch


class AttentionRegistry:
    def __init__(self) -> None:
        self.registry = {
            "scaled_dot_product": self.scaled_dot_product_attention,
            "fast_dot_product": self.fast_dot_product_attention,
            "relative_dot_product": self.relative_dot_product_attention,
            "skipped_relative_dot_product": self.skipped_relative_dot_product_attention,
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
    def relative_dot_product_attention(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        attention_logits = (
            torch.matmul(Q, K.transpose(-1, -2)) + torch.matmul(Q, R.transpose(-1, -2))
        ) / math.sqrt(Q.size(-1))
        attention = torch.nn.functional.softmax(attention_logits, dim=-1)
        return torch.matmul(attention, V)

    @staticmethod
    def skipped_relative_dot_product_attention(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        attention_logits = (
            torch.matmul(Q, K.transpose(-1, -2)) + torch.matmul(Q, R.transpose(-1, -2))
        ) / math.sqrt(Q.size(-1))
        attention = torch.nn.functional.softmax(attention_logits, dim=-1)
        return torch.matmul(attention, V) + Q

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
        pool_attention: bool = False,
        pool_latent_dim: int = 64,
    ):
        super(MultiHeadAttention, self).__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim
        self.n_head = n_head
        self.pool_attention = pool_attention
        self.pool_latent_dim = pool_latent_dim

        if pool_attention and pool_latent_dim >= head_dim:
            raise ValueError(
                f"pool_latent_dim ({pool_latent_dim}) must be less than head_dim ({head_dim})"
            )

        self.Q = torch.nn.Linear(input_dim, head_dim * n_head)
        self.K = torch.nn.Linear(input_dim, head_dim * n_head)
        self.V = torch.nn.Linear(input_dim, head_dim * n_head)

        if pool_attention:
            self.PQ = torch.nn.Conv2d(
                head_dim,
                pool_latent_dim,
                kernel_size=1,
            )
            self.PK = torch.nn.Conv2d(
                head_dim,
                pool_latent_dim,
                kernel_size=1,
            )
            self.PV = torch.nn.Conv2d(
                head_dim,
                pool_latent_dim,
                kernel_size=1,
            )
            self.dispatcher = torch.nn.Linear(pool_latent_dim * n_head, input_dim)
        else:
            self.dispatcher = torch.nn.Linear(head_dim * n_head, input_dim)

        attention_registry = AttentionRegistry()
        self.attention_method = attention_registry.get(attention_method)

        if "skip" or "relative" in attention_method:
            pass

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

        # Pool Attention: Q|K|V[B, N, L, H] -> Q|K|V[B, N, l, H], where l << L
        if self.pool_attention:
            Q = self.PQ(Q.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            K = self.PK(K.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            V = self.PV(V.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # Compute Attention: Q|K|V[B, N, L, H] -> Attention[B, N, L, H]
        attention = self.attention_method(Q, K, V)

        # Concatenate heads: [B, N, L, H] -> [B, L, N * H]
        stacked_attention = attention.transpose(1, 2).contiguous().view(B, L, -1)

        # Project Attention: [B, L, N * H] -> [B, L, D]
        return self.dispatcher(stacked_attention)


head = MultiHeadAttention(512, 64, 8, "fast_dot_product", True, pool_latent_dim=32)
x = torch.randn(4, 64, 512)
y = head(x)
print(y.shape)
