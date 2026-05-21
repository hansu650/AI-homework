"""Formal training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_SIZE, IMAGE_SIZE, LEARNING_RATE
from src.lightning.data_module import NYU5DataModule
from src.lightning.lit_segmentation import LitSegmentation
from src.utils.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CampusDepthSegLite.")
    parser.add_argument("--data_dir", default="data/nyu5")
    parser.add_argument("--train_split", default="data/nyu5/splits/train.txt")
    parser.add_argument("--val_split", default="data/nyu5/splits/val.txt")
    parser.add_argument("--variant", default="rgbd_boundary")
    parser.add_argument("--experiment_name", default="debug")
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", default="32")
    parser.add_argument("--limit_train_batches", type=_parse_batch_limit, default=1.0)
    parser.add_argument("--limit_val_batches", type=_parse_batch_limit, default=1.0)
    parser.add_argument("--image_height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--image_width", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _parse_batch_limit(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, seed_cuda=args.accelerator == "gpu")
    data_dir = _resolve_project_path(args.data_dir)
    train_split = _resolve_project_path(args.train_split)
    val_split = _resolve_project_path(args.val_split)
    run_dir = PROJECT_ROOT / "outputs" / "runs" / args.experiment_name

    data_module = NYU5DataModule(
        train_split=train_split,
        val_split=val_split,
        data_dir=data_dir,
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = LitSegmentation(learning_rate=args.learning_rate, variant=args.variant)

    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = CSVLogger(
        save_dir=PROJECT_ROOT / "outputs" / "runs",
        name=args.experiment_name,
        version="",
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        callbacks=[checkpoint, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=data_module)
    print(f"best checkpoint path: {checkpoint.best_model_path}")
    print(f"best val_mIoU: {checkpoint.best_model_score}")
    print(f"experiment output dir: {run_dir}")


def _resolve_project_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


if __name__ == "__main__":
    main()
