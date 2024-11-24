import pytest
import rootutils


# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(root)
from src.datamodules.dogbreed_modules import DogBreedImageDataModule

@pytest.fixture
def datamodule():
    return DogBreedImageDataModule(dl_path = "data", num_workers = 0, batch_size = 32, splits = [0.8, 0.2],
        pin_memory = False, samples = 5, filenames = [], classes = {})

@pytest.mark.dependency(on=['tests/test_infer.py'])
@pytest.mark.order(4)
def test_dogbreed_datamodule_setup(datamodule):
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    # print("reach")
    # datamodule.setup()
    # datamodule.setup(stage='fit')

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    total_size = (
        len(datamodule.train_dataset)
        + len(datamodule.test_dataset)
    )
    print("total size", total_size)
    # assert total_size == len(datamodule._dataset)

@pytest.mark.dependency(on=['tests/test_infer.py'])
@pytest.mark.order(5)
def test_dogbreed_datamodule_train_val_test_splits(datamodule):
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    assert len(datamodule.train_dataset) > len(datamodule.val_dataset)
    assert len(datamodule.train_dataset) > len(datamodule.test_dataset)

@pytest.mark.dependency(on=['tests/test_infer.py'])
@pytest.mark.order(6)
def test_dogbreed_datamodule_dataloaders(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None