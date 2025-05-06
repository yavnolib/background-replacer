import os

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from background_replacer.utils.transforms import (
    get_train_transforms,
    get_val_transforms,
)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms):
        self.image_paths = sorted(
            [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        )
        self.mask_paths = sorted(
            [os.path.join(mask_dir, x) for x in os.listdir(mask_dir)]
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)
        augmented = self.transforms(image=image, mask=mask)
        return augmented["image"], augmented["mask"].unsqueeze(0)


class SegmentationDataModule(torch.utils.data.DataLoader):
    def __init__(self, image_dir, mask_dir, batch_size=8, num_workers=4):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = SegmentationDataset(
            self.image_dir + "/train/images",
            self.mask_dir + "/train/masks",
            get_train_transforms(),
        )
        self.val_ds = SegmentationDataset(
            self.image_dir + "/val/images",
            self.mask_dir + "/val/masks",
            get_val_transforms(),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
