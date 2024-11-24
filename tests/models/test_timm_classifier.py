import pytest
import torch

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.dogbreed_classifer import DogBreedClassifier

@pytest.mark.dependency(on=['tests/test_infer.py'])
@pytest.mark.order(7)
def test_dogbreed_classifer_forward():
    model = DogBreedClassifier(base_model="resnet18", num_classes=10)
    batch_size, channels, height, width = 4, 3, 224, 224
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    assert output.shape == (batch_size, 10)