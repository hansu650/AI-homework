"""PyTorch Lightning module for CampusDepthSegLite."""

from __future__ import annotations

import torch
from lightning import LightningModule

from src.config import LEARNING_RATE, NUM_CLASSES, WEIGHT_DECAY
from src.models.campus_depthseg_lite import CampusDepthSegLite
from src.utils.losses import SegmentationLoss
from src.utils.metrics import confusion_matrix, mean_iou, pixel_accuracy


class LitSegmentation(LightningModule):
    """Training, validation, and test wrapper for the segmentation model."""

    def __init__(
        self,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = CampusDepthSegLite(num_classes=num_classes)
        self.loss_fn = SegmentationLoss(num_classes=num_classes)
        self.num_classes = num_classes
        self.val_confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.test_confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.model(rgb, depth)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch["rgb"], batch["depth"])
        loss = self.loss_fn(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        logits = self(batch["rgb"], batch["depth"])
        loss = self.loss_fn(logits, batch["label"])
        matrix = confusion_matrix(logits.detach(), batch["label"], self.num_classes)
        self.val_confmat = self.val_confmat.to(matrix.device) + matrix
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_confmat = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.long,
            device=self.device,
        )

    def on_validation_epoch_end(self) -> None:
        self.log("val_mIoU", mean_iou(self.val_confmat), prog_bar=True)
        self.log("val_pixel_acc", pixel_accuracy(self.val_confmat), prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch["rgb"], batch["depth"])
        loss = self.loss_fn(logits, batch["label"])
        matrix = confusion_matrix(logits.detach(), batch["label"], self.num_classes)
        self.test_confmat = self.test_confmat.to(matrix.device) + matrix
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_confmat = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.long,
            device=self.device,
        )

    def on_test_epoch_end(self) -> None:
        self.log("test_mIoU", mean_iou(self.test_confmat))
        self.log("test_pixel_acc", pixel_accuracy(self.test_confmat))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
