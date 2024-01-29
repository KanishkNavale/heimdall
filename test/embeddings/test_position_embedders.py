from unittest import TestCase

import torch

from heimdall.embeddings.position_embedder import (
    PositionalEmbedder1D,
    PositionalEmbedder2D,
    PositionalEmbedder3D,
)


class TestPositionalEmbedder1D(TestCase):
    def test_position_embedder(self) -> None:
        embedder = PositionalEmbedder1D(d_model=64)

        x = torch.randn(2, 100, 64)
        x = embedder.forward(x)

        self.assertEqual(x.shape, (2, 100, 64))


class TestPositionalEmbedder2D(TestCase):
    def test_position_embedder(self) -> None:
        embedder = PositionalEmbedder2D(d_model=64)

        x = torch.randn(2, 100, 100, 64)
        x = embedder.forward(x)

        self.assertEqual(x.shape, (2, 100, 100, 64))


class TestPositionalEmbedder3D(TestCase):
    def test_position_embedder(self) -> None:
        embedder = PositionalEmbedder3D(d_model=64)

        x = torch.randn(2, 100, 100, 100, 64)
        x = embedder.forward(x)

        self.assertEqual(x.shape, (2, 100, 100, 100, 64))
