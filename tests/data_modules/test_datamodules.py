import pytest
import rootutils


# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(root)
from src.datamodules.cat_dog_modules import CatDogImageDataModule

@pytest.fixture
def datamodule():
    return CatDogImageDataModule(dl_path = "data/cats_and_dogs_filtered", num_workers = 0, batch_size = 32, splits = [0.8, 0.1, 0.1],
        pin_memory = False, samples = 5, filenames = [], classes = {0: 'Cat', 1: 'Dog'})

@pytest.mark.dependency(on=['tests/test_infer.py'])
@pytest.mark.order(4)
def test_dogbreed_datamodule_dataloaders(datamodule):
    # datamodule.prepare_data()
    # datamodule.setup()
    datamodule = CatDogImageDataModule(dl_path = "data/cats_and_dogs_filtered", num_workers = 0, batch_size = 32, splits = [0.8, 0.1, 0.1],
        pin_memory = False, samples = 5, filenames = [], classes = {0: 'Cat', 1: 'Dog'})
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    
    total = len(train_loader.dataset)  + len(val_loader.dataset)  + len(test_loader.dataset)
    print("total-", len(train_loader.dataset), len(val_loader.dataset),  len(test_loader.dataset))
    assert total > 0
    
    assert len(train_loader.dataset)/total >= (0.5), f"Train dataset length mismatch: Expected 50%, Got {(len(train_loader.dataset)/total)*100}%"
    assert len(val_loader.dataset)/total >= (0.25-0.01), f"Validation dataset length mismatch: Expected 25%, Got {(len(val_loader.dataset)/total)*100}%"
    assert len(test_loader.dataset)/total >= (0.25-0.01), f"Test dataset length mismatch: Expected 25%, Got {(len(test_loader.dataset)/total)*100}%"