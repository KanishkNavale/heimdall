from typing import Optional, Tuple

import torch

from heimdall.embeddings.position_embedder import PositionalEmbedder2D


class AttentionPooler(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        has_cls_token: bool = True,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.pooler = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.has_cls_token = has_cls_token

    def forward(
        self, x: torch.Tensor, hw_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Check shapes
        H, W = hw_shape
        B, N, _, C = x.shape

        if self.has_cls_token:
            cls_tokens = x[:, :, 0, :].unsqueeze(2)
            x = x[:, :, 1:, :]

        # Reshape: X[B, N, L, C] -> X[B * N, C, H, W]
        x = x.transpose(1, 2).reshape(B * N, C, H, W)

        # Pooler: X[B * N, C, H, W] -> X[B * N, D, h, w], where h << H, w << W, C <= D
        pooled_features = self.pooler.forward(x)
        hw_shape = (pooled_features.shape[-2], pooled_features.shape[-1])

        # Reshape: X[B * N, D, h, w] -> X[B, N, D, h, w]
        reshaped_pooled_features = pooled_features.reshape(
            B,
            N,
            pooled_features.shape[1],
            pooled_features.shape[2],
            pooled_features.shape[3],
        )

        # Flatten: X[B, N, D, h, w] -> X[B, N, h * w, D]
        flattened_pooled_features = reshaped_pooled_features.flatten(-2).transpose(
            -2, -1
        )

        # Add cls token: X[B, N, h * w, D] -> X[B, N, 1 + (h * w), D]
        if self.has_cls_token:
            flattened_pooled_features = torch.cat(
                [cls_tokens, flattened_pooled_features], dim=2
            )

        return flattened_pooled_features, hw_shape


class MultiHeadPooledSelfAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int = 512,
        n_head: int = 8,
        has_cls_token: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim
        self.n_head = n_head
        self.has_cls_token = has_cls_token

        self.Q = torch.nn.Linear(input_dim, head_dim * n_head)
        self.K = torch.nn.Linear(input_dim, head_dim * n_head)
        self.V = torch.nn.Linear(input_dim, head_dim * n_head)

        self.PQ = AttentionPooler(input_dim, head_dim, has_cls_token=has_cls_token)
        self.PK = AttentionPooler(input_dim, head_dim, has_cls_token=has_cls_token)
        self.PV = AttentionPooler(input_dim, head_dim, has_cls_token=has_cls_token)

        # Rotary Embeddings instrad of Positional Embeddings
        self.relative_pos_embedder = PositionalEmbedder2D(head_dim)

        self.dispatcher = torch.nn.Linear(head_dim * n_head, input_dim)

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

    def _add_relative_pos_encoding(
        self,
        k: torch.Tensor,
        hw_shape: Tuple[int, int],
    ) -> torch.Tensor:
        B, N, L, C = k.shape
        h, w = hw_shape

        if self.has_cls_token:
            cls_tokens = k[:, :, 0, :].unsqueeze(2)
            k = k[:, :, 1:, :]

        # Reshape: X[B, N, L, C] -> X[B * N, h, w, C]
        k = k.reshape(B * N, h, w, C)

        embeddings = self.relative_pos_embedder(k)
        k = k + embeddings

        # Reshape: X[B * N, h, w, C] -> X[B, N, L, C]
        k = k.reshape(B, N, L - 1, C)

        if self.has_cls_token:
            embeddings = torch.concat([cls_tokens, k], dim=-2)

        return embeddings

    def forward(
        self, x: torch.Tensor, hw_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, L, _ = x.size()

        # X[B: Batch Size, L: Sequence Length, D: Embedding Dim.] -> Q|K|V[B, L, N: No. of Heads * H:]
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # Q|K|V[B, L, N * H] -> Q|K|V[B, N, L, H]
        Q = Q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        # Pooled Features: Q|K|V[B, N, L, H] -> Pooled Features[B, N, l, h], where l <= L, h <= H
        PQ, pq_shape = self.PQ.forward(Q, hw_shape)
        PK, _ = self.PK.forward(K, hw_shape)
        PV, _ = self.PV.forward(V, hw_shape)

        # Add Relative Positional Embeddings
        PQ = self._add_relative_pos_encoding(PQ, pq_shape)

        # Compute Attention: PQ|PK|PV[B, N, l, h] -> Attention[B, N, l, h]
        attention_logits = torch.matmul(PQ, PK.transpose(-2, -1)) / (self.head_dim**0.5)
        attention = torch.nn.functional.softmax(attention_logits, dim=-1)
        self_attention = torch.matmul(attention, PV)

        if self.has_cls_token:
            self_attention[:, :, 1:, :] += Q[:, :, 1:, :]
            skipped_attention = self_attention
        else:
            skipped_attention = self_attention + Q

        # Concatenate heads: [B, N, l, h] -> [B, l, N * h]
        stacked_attention = (
            skipped_attention.transpose(2, 1).contiguous().view(B, L, -1)
        )

        # Project: [B, l, N * h] -> [B, l, D]
        return self.dispatcher(stacked_attention), pq_shape
