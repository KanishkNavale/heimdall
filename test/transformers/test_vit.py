from unittest import TestCase

import torch

from heimdall.transformers.vision.vit import MViT, ViT


class TestBaseViT(TestCase):
    def test_forward_pass(self) -> None:
        B, C, H, W, D = 2, 3, 224, 224, 768
        x = torch.randn(B, C, H, W)

        model = ViT(
            in_channels=C,
            out_channels=768,
            patch_size=16,
            n_attention_heads=4,
            n_encoder_layers=12,
        )

        output = model.forward(x)

        lenght = (224 // 16) ** 2 + 1  # 1: CLS token

        self.assertEqual(output.shape, (B, lenght, D))

    def test_classification_token_forward(self) -> None:
        B, C, H, W, D = 2, 3, 224, 224, 768
        x = torch.randn(B, C, H, W)

        model = ViT(
            in_channels=C,
            out_channels=768,
            patch_size=16,
            n_attention_heads=4,
            n_encoder_layers=12,
        )

        output = model.compute_classification_logits(x)

        self.assertEqual(output.shape, (B, D))


class TestMViTS(TestCase):
    def test_forward_pass(self) -> None:
        B, C, T, H, W = 2, 3, 1, 224, 224
        x = torch.randn(B, C, T, H, W)

        model = MViT(
            in_channels=C,
            out_channels=128,
            patch_size=4,
            temporal_size=1,
            n_heads=4,
            block_config=[3, 7, 6],
        )

        feature = model.forward(x)

        self.assertEqual(feature.shape, (B, 1, 128 * 64))
