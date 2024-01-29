from unittest import TestCase

import torch

from heimdall.embeddings.patch_embedder import PatchEmbedder, TemporalPatchEmbedder


class TestPatchEmbedder(TestCase):
    def test_patch_embedder(self) -> None:
        patch_embedder = PatchEmbedder(
            in_channels=3, out_channels=64, patch_size=16, add_cls_token=True
        )

        x = torch.randn(1, 3, 256, 256)
        x = patch_embedder.forward(x)

        n_patches = (256 // 16) ** 2 + 1  # 1 for the cls token

        self.assertEqual(x.shape, (1, n_patches, 64))

    def test_patch_embedder_no_cls_token(self) -> None:
        patch_embedder = PatchEmbedder(
            in_channels=3, out_channels=64, patch_size=16, add_cls_token=False
        )

        x = torch.randn(1, 3, 256, 256)
        x = patch_embedder.forward(x)

        n_patches = (256 // 16) ** 2

        self.assertEqual(x.shape, (1, n_patches, 64))

    def test_patch_embedder_channel_first(self) -> None:
        patch_embedder = PatchEmbedder(
            in_channels=3,
            out_channels=64,
            patch_size=16,
            add_cls_token=True,
            channel_last=False,
        )

        x = torch.randn(1, 3, 256, 256)
        x = patch_embedder(x)

        n_patches = (256 // 16) ** 2 + 1

        self.assertEqual(x.shape, (1, 64, n_patches))

    def test_patch_embedder_return_shape(self) -> None:
        patch_embedder = PatchEmbedder(
            in_channels=3, out_channels=64, patch_size=16, add_cls_token=True
        )

        x = torch.randn(1, 3, 256, 256)
        _, shape = patch_embedder(x, extract_pre_flat_shape=True)

        patch_size = 256 // 16

        self.assertEqual(shape, (patch_size, patch_size))


class TestTemporalPatchEmbedder(TestCase):
    def test_temporal_patch_embedder(self) -> None:
        temporal_patch_embedder = TemporalPatchEmbedder(
            in_channels=3, out_channels=64, patch_size=16, temporal_size=2
        )

        x = torch.randn(1, 3, 16, 256, 256)
        x = temporal_patch_embedder.forward(x)

        n_temporal_patches = ((256 // 16) ** 2) * (16 // 2) + 1  # 1 for the cls token

        self.assertEqual(x.shape, (1, n_temporal_patches, 64))

    def test_temporal_patch_embedder_no_cls_token(self) -> None:
        temporal_patch_embedder = TemporalPatchEmbedder(
            in_channels=3,
            out_channels=64,
            patch_size=16,
            temporal_size=2,
            add_cls_token=False,
        )

        x = torch.randn(1, 3, 16, 256, 256)
        x = temporal_patch_embedder.forward(x)

        self.assertEqual(x.shape, (1, 2048, 64))

    def test_temporal_patch_embedder_channel_first(self) -> None:
        temporal_patch_embedder = TemporalPatchEmbedder(
            in_channels=3,
            out_channels=64,
            patch_size=16,
            temporal_size=2,
            add_cls_token=True,
            channel_last=False,
        )

        x = torch.randn(1, 3, 16, 256, 256)
        x = temporal_patch_embedder(x)

        self.assertEqual(x.shape, (1, 64, 2049))

    def test_temporal_patch_embedder_return_shape(self) -> None:
        temporal_patch_embedder = TemporalPatchEmbedder(
            in_channels=3, out_channels=64, patch_size=16, temporal_size=2
        )

        x = torch.randn(1, 3, 16, 256, 256)
        _, shape = temporal_patch_embedder(x, extract_pre_flat_shape=True)

        patch_shape = (16 // 2, 256 // 16, 256 // 16)

        self.assertEqual(shape, patch_shape)
