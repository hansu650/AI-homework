"""Single RGB-D image prediction entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGE_SIZE
from src.datasets.transforms import depth_to_tensor, resize_sample, rgb_to_tensor
from src.lightning.lit_segmentation import LitSegmentation
from src.utils.visualization import colorize_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one RGB-D image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "figures" / "prediction_mask.png"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", default="rgbd_boundary")
    parser.add_argument("--image_height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--image_width", type=int, default=IMAGE_SIZE[1])
    return parser.parse_args()


def validate_file(path: str, name: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{name} not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"{name} is not a file: {file_path}")
    return file_path


def validate_device(device: str) -> torch.device:
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is not available: {device}")
    return selected


def main() -> None:
    args = parse_args()
    checkpoint = validate_file(args.checkpoint, "Checkpoint")
    rgb_path = validate_file(args.rgb, "RGB image")
    depth_path = validate_file(args.depth, "Depth image")
    device = validate_device(args.device)

    rgb_image = Image.open(rgb_path).convert("RGB")
    depth_image = Image.open(depth_path)
    dummy_label = Image.fromarray(np.zeros((rgb_image.height, rgb_image.width), dtype=np.uint8))
    rgb_image, depth_image, _ = resize_sample(
        rgb_image,
        depth_image,
        dummy_label,
        size=(args.image_height, args.image_width),
    )

    rgb = rgb_to_tensor(rgb_image).unsqueeze(0).to(device)
    depth = depth_to_tensor(depth_image).unsqueeze(0).to(device)

    model = LitSegmentation.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
        variant=args.variant,
    )
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(rgb, depth)
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, colorize_mask(mask))
    print(f"prediction saved: {output_path}")


if __name__ == "__main__":
    main()
