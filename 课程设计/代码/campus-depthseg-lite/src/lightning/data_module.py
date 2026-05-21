"""Lightning data module for NYU5 splits."""

from __future__ import annotations

from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS
from src.datasets.nyu5_dataset import NYU5Dataset


class NYU5DataModule(LightningDataModule):
    """Build train, validation, and test dataloaders from split files."""

    def __init__(
        self,
        train_split: str | Path | None = None,
        val_split: str | Path | None = None,
        test_split: str | Path | None = None,
        data_dir: str | Path | None = None,
        image_size: tuple[int, int] = IMAGE_SIZE,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ) -> None:
        super().__init__()
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self.train_split is None:
                raise ValueError("train_split is required for training")
            if self.val_split is None:
                raise ValueError("val_split is required for validation")
            self.train_dataset = NYU5Dataset(
                self.train_split,
                data_dir=self.data_dir,
                image_size=self.image_size,
                training=True,
            )
            self.val_dataset = NYU5Dataset(
                self.val_split,
                data_dir=self.data_dir,
                image_size=self.image_size,
                training=False,
            )
        if stage in (None, "test"):
            if self.test_split is None:
                raise ValueError("test_split is required for testing")
            self.test_dataset = NYU5Dataset(
                self.test_split,
                data_dir=self.data_dir,
                image_size=self.image_size,
                training=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
