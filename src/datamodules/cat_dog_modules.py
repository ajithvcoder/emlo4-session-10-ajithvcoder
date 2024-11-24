from pathlib import Path
from typing import Union, List
import os

import lightning as L
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

class CustomImageFolder(ImageFolder):
    def __init__(self, root, filenames, transform=None):
        super().__init__(root, transform=transform)
        # Filter the dataset based on the provided filenames
        # self.imgs = [os.path.basename(file) for file in filenames]
        self.infer_imgs = [img for img in self.imgs if os.path.basename(img[0]) in filenames]
        # print("self.imgs")
        # print(self.infer_imgs)
        self.imgs = self.infer_imgs
        # print("length-", self.imgs)
        # self.length = 5

    def __getitem__(self, index):
        
        # print("self.infer_imgs-", self.infer_imgs[index])
        path, target = self.imgs[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path
    
    def __len__(self):
        return len(self.imgs)


class CatDogImageDataModule(L.LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 32, splits: List = [0.8, 0.2],
        pin_memory: bool = False, samples: int = 5, filenames: List = [], classes: dict = {}):
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._splits = splits
        self._pin_memory = pin_memory
        self._samples = samples
        self._filenames = filenames
        print("reach catdog module ...")
    # def prepare_data(self):
    #     """Download images and prepare images datasets."""
    #     # print("val",self.data_path.joinpath("val"))
    #     pass
    
    # def setup(self, stage: str = None):
    #     pass

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def create_infer_dataset(self, root, filenames, transform):
        return CustomImageFolder(root=root, filenames=filenames, transform=transform)

    def __dataloader(self, train: bool = False, test: bool = False, infer: bool = False):
        """Train/validation/test loaders."""

        if train:
            dataset = self.create_dataset(self.data_path.joinpath("train"), self.train_transform)
        elif test:
            dataset = self.create_dataset(self.data_path.joinpath("test"), self.valid_transform)
        elif infer:
            dataset = self.create_infer_dataset(self.data_path.joinpath("test"), self._filenames, self.valid_transform)
            self._batch_size =  1
        print(len(dataset), "--train length", "batch size", self._batch_size)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        store = self.__dataloader(train=True)
        print("store", len(store))
        return store

    def val_dataloader(self):
        return self.__dataloader(test=True)

    def test_dataloader(self):
        return self.__dataloader(test=True)  # Using validation dataset for testing

    def predict_dataloader(self):
        self._batch_size =  1
        return self.__dataloader(infer=True)  # Using validation dataset for testing