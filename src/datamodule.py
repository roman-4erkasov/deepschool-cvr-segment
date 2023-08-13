import os
import sys
import typing as tp
import torchvision as thv
from src.config import DataConfig
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.utils import split_3way, ToTensor, Compose, Resize
import pandas as pd
from src.dataset import Dataset
import albumentations as albu

class BarcodeDM(LightningDataModule):
    """
    DataModule for data "ru-goods-barcodes" from 
    https://www.kaggle.com/datasets/kniazandrew/ru-goods-barcodes
    """
    def __init__(self, config: DataConfig, task, dry_run: bool = False):
        super().__init__()
        self.batch_size = config.batch_size
        self.n_workers = config.n_workers
        self.train_size = config.train_size
        self.val_size = config.val_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_path = config.data_path
        self.scopes_path = config.scopes_path
        self.images_path = config.images_path
        self.task = task
        self.dry_run = dry_run
        
        self.ann_train = os.path.join(self.scopes_path, "train.tsv")
        self.ann_val = os.path.join(self.scopes_path, "val.tsv")
        self.ann_test = os.path.join(self.scopes_path, "test.tsv")
        # self._train_augs = get_train_augmentation(config.img_width, config.img_height)
        # self._test_augs = get_val_augmentation(config.img_width, config.img_height)

    def prepare_data(self):
        if not os.path.exists(self.data_path):
            raise Exception(
                f"Folder not found: \"{self.data_path}\". "
                "Please download dataset from "
                "https://www.kaggle.com/"
                "datasets/kniazandrew/ru-goods-barcodes "
                "and set to the folder above."
            )
        if (
            os.path.join(self.scopes_path, "train.tsv")
            and
            os.path.join(self.scopes_path, "train.tsv")
            and
            os.path.join(self.scopes_path, "train.tsv")
        ):
            return
        split_3way(
            self.ann_path, 
            self.scopes_path,
            train_prop=self.train_size,
            val_prop=self.val_size,
        )

    def setup(self, stage: tp.Optional[str] = None):
        """
        """
        self.train_dataset = Dataset(
            ann_path=self.ann_train, 
            images_path=self.images_path,
            transforms=self.get_transforms(),
            dry_run=self.dry_run,
            task=self.task,
        )
        self.val_dataset = Dataset(
            ann_path=self.ann_train if self.dry_run else self.ann_val, 
            images_path=self.images_path,
            transforms=self.get_transforms(),
            dry_run=self.dry_run,
            task=self.task,
        )
        self.test_dataset = Dataset(
            ann_path=self.ann_train if self.dry_run else self.ann_test, 
            images_path=self.images_path,
            transforms=self.get_transforms(),
            dry_run=self.dry_run,
            task=self.task,
        )

    def get_collate_fn(self):
        if self.task=="detection":
            return lambda batch: tuple(zip(*batch))
        elif self.task=="segmentation":
            return None
        else:
            raise Exception()

    def get_transforms(self):
        if self.task=="detection":
            # return Compose(
            #     [ToTensor()]
            # )
            # return get_transforms()
            return albu.Compose(
                [
                    albu.Resize(32*20, 32*16),
                ],
                bbox_params=albu.BboxParams(format="pascal_voc")
            )
        elif self.task=="segmentation":
            # return albu.Compose(
            #     # [albu.Resize(32*40, 32*33)]
            #     [albu.Resize(32*20, 32*16)]
            # )
            return None
        else:
            raise Exception()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.get_collate_fn()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.get_collate_fn()
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.get_collate_fn()
        )
