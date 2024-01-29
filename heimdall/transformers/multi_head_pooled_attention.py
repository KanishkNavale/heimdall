from typing import Optional, Tuple

import torch

from heimdall.embeddings.position_embedder import PositionalEmbedder3D


def _condition_hw_to_thw_shape(
    thw_shape: Tuple[int, int] | Tuple[int, int, int],
) -> Tuple[int, int, int]:
    return (1, *thw_shape) if len(thw_shape) == 2 else thw_shape


class AttentionPooler(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        patch_size: int = 2,
        temporal_stride: int = 1,
        has_cls_token: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels * 2  # MViT's default scaling factor

        self.pooler = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(temporal_stride, patch_size, patch_size),
            stride=(temporal_stride, patch_size, patch_size),
            bias=bias,
        )

        self.has_cls_token = has_cls_token

    def forward(
        self, x: torch.Tensor, thw_shape: Tuple[int, int] | Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Check shapes

        T, H, W = _condition_hw_to_thw_shape(thw_shape)

        # B: Batch Size, N: No. of Heads, L: Sequence Length, D: Embedding Dim.
        B, N, _, D = x.shape

        if self.has_cls_token:
            cls_tokens = x[:, :, 0, :].unsqueeze(2)
            x = x[:, :, 1:, :]

        # Reshape: X[B, N, L, D] -> X[B * N, D, T, H, W]
        x = x.transpose(1, 2).reshape(B * N, D, T, H, W)

        # Pooler: X[B * N, D, T, H, W] -> X[B * N, D, t, h, w], where h << H, w << W, , t<<T, C <= D
        pooled_features = self.pooler.forward(x)
        pooled_thw_shape = pooled_features.shape[-3:]

        # Reshape: X[B * N, D, t, h, w] -> X[B, N, D, t, h, w]
        reshaped_pooled_features = pooled_features.reshape(
            B, N, pooled_features.shape[1], *pooled_thw_shape
        )

        # Flatten: X[B, N, D, t, h, w] -> X[B, N, t * h * w, D]
        flattened_pooled_features = reshaped_pooled_features.flatten(3).transpose(
            -2, -1
        )

        # Add cls token: X[B, N, h * w, D] -> X[B, N, 1 + (h * w), D]
        if self.has_cls_token:
            flattened_pooled_features = torch.cat(
                [cls_tokens, flattened_pooled_features], dim=2
            )

        return flattened_pooled_features, pooled_thw_shape


class MultiHeadPooledAttention(torch.nn.Module):
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
        self.PX = AttentionPooler(input_dim, head_dim, has_cls_token=has_cls_token)

        self.output_layer_norm = torch.nn.LayerNorm(input_dim)

        # Rotary Embeddings instrad of Positional Embeddings
        self.relative_pos_embedder = PositionalEmbedder3D(head_dim)

        self.dispatcher = torch.nn.Linear(head_dim * n_head, input_dim)

        self.attention_scale = head_dim**-0.5

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

    def _add_relative_pos_attention(
        self,
        attention_logits: torch.Tensor,
        Q: torch.Tensor,
        thw_shape: Tuple[int, int],
    ) -> torch.Tensor:
        T, H, W = _condition_hw_to_thw_shape(thw_shape)
        B, N, _, D = Q.shape

        cls_index = 1 if self.has_cls_token else 0

        # Reshape: X[B, N, L, D] -> X[B*N, T, H, W, D]
        q = Q[:, :, cls_index:].reshape(B * N, T, H, W, D)

        # Compute relative positional embeddings
        embeddings = self.relative_pos_embedder(q)

        qr = torch.matmul(
            q.reshape(B * N, T * H * W, D),
            embeddings.reshape(B * N, T * H * W, D).transpose(-2, -1),
        )

        # Reshape: X[B * N, T, H, W] -> X[B, N, T*H*W]
        attention_logits[:, :, cls_index:, cls_index:] += qr.reshape(
            B, N, T * H * W, T * H * W
        )
        return attention_logits

    def forward(
        self, x: torch.Tensor, thw_shape: Tuple[int, int] | Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, L, _ = x.size()
        thw_shape = _condition_hw_to_thw_shape(thw_shape)

        # X[B: Batch Size, L: Sequence Length, D: Embedding Dim.] -> Q|K|V[B, L, N: No. of Heads * D]
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # Q|K|V[B, L, N * D] -> Q|K|V[B, N, L, D]
        Q = Q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        # Pooled Features: Q|K|V[B, N, L, D] -> Pooled Features[B, N, l, h], where l <= L, d <= D
        PQ, pq_shape = self.PQ.forward(Q, thw_shape)
        PK, _ = self.PK.forward(K, thw_shape)
        PV, _ = self.PV.forward(V, thw_shape)

        # Compute Attention: PQ|PK|PV[B, N, l, d] -> Attention[B, N, l, d]
        attention_logits = torch.matmul(PQ, PK.transpose(-2, -1))
        scaled_attention_logits = attention_logits * self.attention_scale
        relative_attention_logits = self._add_relative_pos_attention(
            scaled_attention_logits, PQ, pq_shape
        )
        attention = torch.nn.functional.softmax(relative_attention_logits, dim=-1)
        residual_self_attention = torch.matmul(attention, PV) + PQ

        if self.has_cls_token:
            residual_self_attention[:, :, 1:, :] += PQ[:, :, 1:, :]
            skipped_attention = residual_self_attention
        else:
            skipped_attention = residual_self_attention + PQ

        # Concatenate heads: [B, N, l, d] -> [B, l, N * d]
        stacked_attention = skipped_attention.transpose(1, 2).contiguous().flatten(2)

        # Compute Pooled Input: X[B, L, D] -> [B, l, d]
        PX, _ = self.PX.forward(x.unsqueeze(dim=1), thw_shape)
        PX = PX.squeeze(dim=1)

        # Project: [B, l, N * d] -> [B, l, D]
        skipped_projected_features = self.dispatcher(stacked_attention) + PX
        return self.output_layer_norm(skipped_projected_features), pq_shape
