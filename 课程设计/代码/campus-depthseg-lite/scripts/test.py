"""Formal test entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lightning import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS
from src.lightning.data_module import NYU5DataModule
from src.lightning.lit_segmentation import LitSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CampusDepthSegLite.")
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--image_height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--image_width", type=int, default=IMAGE_SIZE[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_module = NYU5DataModule(
        test_split=args.test_split,
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = LitSegmentation.load_from_checkpoint(str(checkpoint))
    trainer = Trainer(accelerator=args.accelerator, devices=args.devices)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
