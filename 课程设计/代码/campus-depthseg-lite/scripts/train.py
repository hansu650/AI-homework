"""Formal training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_SIZE, IMAGE_SIZE, LEARNING_RATE, NUM_WORKERS
from src.lightning.data_module import NYU5DataModule
from src.lightning.lit_segmentation import LitSegmentation
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CampusDepthSegLite.")
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--val_split", required=True)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--image_height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--image_width", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, seed_cuda=args.accelerator == "gpu")

    data_module = NYU5DataModule(
        train_split=args.train_split,
        val_split=args.val_split,
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = LitSegmentation(learning_rate=args.learning_rate)

    checkpoint = ModelCheckpoint(
        dirpath=PROJECT_ROOT / "outputs" / "checkpoints",
        filename="campus-depthseg-lite-{epoch:02d}-{val_mIoU:.3f}",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
    )
    logger = CSVLogger(save_dir=PROJECT_ROOT / "outputs" / "logs", name="train")

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        callbacks=[checkpoint],
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
