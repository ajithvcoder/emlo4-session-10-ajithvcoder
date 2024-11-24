import pytest
import torch

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.cat_dog_classifier import CatDogClassifier

# @pytest.mark.dependency(on=['tests/test_infer.py'])
# @pytest.mark.order(7)
# def test_dogbreed_classifer_forward():
#     model = DogBreedClassifier(base_model="resnet18", num_classes=10)
#     batch_size, channels, height, width = 4, 3, 224, 224
#     x = torch.randn(batch_size, channels, height, width)
#     output = model(x)
#     assert output.shape == (batch_size, 10)
# model:
#   base_model: 'convnext_tiny'
#   pretrained: False
#   num_classes: 2
#   depths: (2, 2, 4, 2)
#   dims: (16, 32, 64, 128)
#   patch_size: 16
#   embed_dim: 128
#   depth: 6
#   num_heads: 8
#   mlp_ratio: 3
#   lr: 1e-3
@pytest.mark.dependency(on=['tests/test_infer.py'])
@pytest.mark.order(7)
def test_classifer_forward():
    model = CatDogClassifier(base_model="convnext_tiny",pretrained=False, num_classes=2, depths="(2, 2, 4, 2)",
                             dims="(16, 32, 64, 128)", patch_size=16, embed_dim=128, depth=6, num_heads=8, mlp_ratio=3, lr=1e-3)
    batch_size, channels, height, width = 4, 3, 224, 224
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    assert output.shape == (batch_size, 2)
