import math

import torch


def sine_embed(x: torch.Tensor) -> torch.Tensor:
    embedding = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
    return torch.flatten(embedding, -2, -1)


class PositionalEmbedder1D(torch.nn.Module):
    def __init__(self, d_model: int):
        super(PositionalEmbedder1D, self).__init__()

        self.normalizer = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(1e4) / d_model)
        )

        self.register_buffer("cached_embedding", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) != 3:
            raise ValueError(
                f"Input tensor must be 3D, but got {len(x.size())}D tensor"
            )

        B, L, D = x.size()

        if self.cached_embedding is not None and self.cached_embedding.shape == x.shape:
            return self.cached_embedding

        if x.size(-1) != D:
            raise ValueError(
                f"Input feature dimension ({x.size(2)}) != to the dimension of positional embedding ({D})"
            )

        positions = torch.arange(L, device=x.device, dtype=self.normalizer.dtype)
        normalized_positions = positions.unsqueeze(dim=-1) * self.normalizer

        embedding = torch.zeros(L, D, device=x.device, dtype=x.dtype)
        embedding[:, :D] = sine_embed(normalized_positions)

        self.cached_embedding = embedding[None, :, :].tile(B, 1, 1)
        return self.cached_embedding


class PositionalEmbedder2D(torch.nn.Module):
    def __init__(self, d_model: int):
        super(PositionalEmbedder2D, self).__init__()

        self.normalizer = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(1e4) / d_model)
        )

        self.register_buffer("cached_embedding", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) != 4:
            raise ValueError(
                f"Input tensor must be 4D, but got {len(x.size())}D tensor"
            )

        B, H, W, D = x.size()

        if self.cached_embedding is not None and self.cached_embedding.shape == x.shape:
            return self.cached_embedding

        if x.size(-1) != D:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) != to the dimension of positional embedding ({D})"
            )

        pos_h = torch.arange(H, device=x.device, dtype=self.normalizer.dtype)
        norm_pos_h = pos_h.unsqueeze(dim=-1) * self.normalizer
        embed_h = torch.zeros(H, D, device=x.device, dtype=x.dtype)
        embed_h[:, :D] = sine_embed(norm_pos_h)

        pos_w = torch.arange(H, device=x.device, dtype=self.normalizer.dtype)
        norm_pos_w = pos_w.unsqueeze(dim=-1) * self.normalizer
        embed_w = torch.zeros(W, D, device=x.device, dtype=x.dtype)
        embed_w[:, :D] = sine_embed(norm_pos_w)

        embedding = torch.zeros(H, W, D * 2, device=x.device, dtype=x.dtype)
        embedding[:, :, :D] = embed_h.unsqueeze(dim=1)
        embedding[:, :, D : 2 * D] = embed_w

        self.cached_embedding = embedding[None, :, :, :D].tile(B, 1, 1, 1)
        return self.cached_embedding


class PositionalEmbedder3D(torch.nn.Module):
    def __init__(self, d_model: int):
        super(PositionalEmbedder3D, self).__init__()

        self.normalizer = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(1e4) / d_model)
        )

        self.register_buffer("cached_embedding", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) != 5:
            raise ValueError(
                f"Input tensor must be 5D, but got {len(x.size())}D tensor"
            )

        B, T, H, W, D = x.size()

        if self.cached_embedding is not None and self.cached_embedding.shape == x.shape:
            return self.cached_embedding

        if x.size(-1) != D:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) != to the dimension of positional embedding ({D})"
            )

        pos_t = torch.arange(T, device=x.device, dtype=self.normalizer.dtype)
        norm_pos_t = pos_t.unsqueeze(dim=-1) * self.normalizer
        embed_t = torch.zeros(T, D, device=x.device, dtype=x.dtype)
        embed_t[:, :D] = sine_embed(norm_pos_t)

        pos_h = torch.arange(H, device=x.device, dtype=self.normalizer.dtype)
        norm_pos_h = pos_h.unsqueeze(dim=-1) * self.normalizer
        embed_h = torch.zeros(H, D, device=x.device, dtype=x.dtype)
        embed_h[:, :D] = sine_embed(norm_pos_h)

        pos_w = torch.arange(H, device=x.device, dtype=self.normalizer.dtype)
        norm_pos_w = pos_w.unsqueeze(dim=-1) * self.normalizer
        embed_w = torch.zeros(W, D, device=x.device, dtype=x.dtype)
        embed_w[:, :D] = sine_embed(norm_pos_w)

        embedding = torch.zeros(T, H, W, D * 3, device=x.device, dtype=x.dtype)
        embedding[:, :, :, :D] = embed_t.unsqueeze(dim=1).unsqueeze(dim=2)
        embedding[:, :, :, D : 2 * D] = embed_h.unsqueeze(dim=1)
        embedding[:, :, :, 2 * D : 3 * D] = embed_w

        self.cached_embedding = embedding[:, :, :, :D].tile(B, 1, 1, 1, 1)
        return self.cached_embedding
