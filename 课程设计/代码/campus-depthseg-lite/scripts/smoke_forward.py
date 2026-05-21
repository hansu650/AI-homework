"""CPU-friendly forward smoke test for CampusDepthSegLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.campus_depthseg_lite import CampusDepthSegLite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one synthetic forward pass.")
    parser.add_argument("--device", default="cpu", help="Device, e.g. cpu or cuda:0.")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--variant", default="rgbd_boundary")
    return parser.parse_args()


def validate_device(device: str) -> torch.device:
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is not available: {device}")
    return selected


def main() -> None:
    args = parse_args()
    device = validate_device(args.device)

    model = CampusDepthSegLite(variant=args.variant).to(device)
    model.eval()

    rgb = torch.rand(2, 3, args.height, args.width, device=device)
    depth = torch.rand(2, 1, args.height, args.width, device=device)

    with torch.no_grad():
        logits = model(rgb, depth)

    param_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"variant: {args.variant}")
    print(f"parameters: {param_count:,}")
    print(f"output_shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
