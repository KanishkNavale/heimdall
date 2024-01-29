from unittest import TestCase

import torch

from heimdall.transformers.vision.vit import BaseViTFeature


class TestBaseViT(TestCase):
    def test_forward_pass(self) -> None:
        B, C, H, W, D = 2, 3, 224, 224, 768
        x = torch.randn(B, C, H, W)

        model = BaseViTFeature(
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

        model = BaseViTFeature(
            in_channels=C,
            out_channels=768,
            patch_size=16,
            n_attention_heads=4,
            n_encoder_layers=12,
        )

        output = model.compute_classification_logits(x)

        self.assertEqual(output.shape, (B, D))
