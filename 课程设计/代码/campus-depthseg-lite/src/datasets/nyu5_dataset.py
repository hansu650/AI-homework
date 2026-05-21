"""RGB-D semantic segmentation dataset for exported NYU5 files."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import IGNORE_INDEX, IMAGE_SIZE, NYU40_TO_5
from src.datasets.transforms import (
    depth_to_tensor,
    horizontal_flip,
    label_to_tensor,
    resize_sample,
    rgb_to_tensor,
)


class NYU5Dataset(Dataset):
    """Read RGB, depth, and five-class labels from split files."""

    def __init__(
        self,
        split_file: str | Path,
        data_dir: str | Path | None = None,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        training: bool = False,
        flip_prob: float = 0.5,
        label_mode: str = "nyu5",
    ) -> None:
        self.split_file = Path(split_file)
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        if not self.split_file.is_file():
            raise ValueError(f"Split path is not a file: {self.split_file}")

        self.root_dir = self._resolve_data_dir(data_dir)
        self.image_size = image_size
        self.training = training
        self.flip_prob = flip_prob
        self.label_mode = label_mode
        if self.label_mode not in {"nyu5", "nyu40"}:
            raise ValueError("label_mode must be 'nyu5' or 'nyu40'")
        self.samples = self._read_split(self.split_file)

        if len(self.samples) == 0:
            raise ValueError(f"Split file contains no samples: {self.split_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        rgb_path, depth_path, label_path = self.samples[index]
        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)
        label = Image.open(label_path)

        rgb, depth, label = resize_sample(rgb, depth, label, self.image_size)
        if self.training and random.random() < self.flip_prob:
            rgb, depth, label = horizontal_flip(rgb, depth, label)

        label_tensor = label_to_tensor(label)
        if self.label_mode == "nyu5":
            mapped_label = self._validate_nyu5_label(label_tensor)
        else:
            mapped_label = self._map_nyu40_label(label_tensor)

        return {
            "rgb": rgb_to_tensor(rgb),
            "depth": depth_to_tensor(depth),
            "label": mapped_label,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "label_path": str(label_path),
        }

    def _read_split(self, split_file: Path) -> List[Tuple[Path, Path, Path]]:
        samples: List[Tuple[Path, Path, Path]] = []
        lines = split_file.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid split line {line_number} in {split_file}: "
                    "expected 'rgb_path depth_path label_path'."
                )
            rgb_path = self._resolve_existing(parts[0], "RGB")
            depth_path = self._resolve_existing(parts[1], "depth")
            label_path = self._resolve_existing(parts[2], "label")
            samples.append((rgb_path, depth_path, label_path))
        return samples

    def _resolve_existing(self, raw_path: str, kind: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.root_dir / path
        if not path.exists():
            raise FileNotFoundError(f"{kind} file not found: {path}")
        if not path.is_file():
            raise ValueError(f"{kind} path is not a file: {path}")
        return path

    def _resolve_data_dir(self, data_dir: str | Path | None) -> Path:
        if data_dir is not None:
            root_dir = Path(data_dir)
        elif self.split_file.parent.name == "splits":
            root_dir = self.split_file.parent.parent
        else:
            root_dir = self.split_file.parent

        if not root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {root_dir}")
        if not root_dir.is_dir():
            raise ValueError(f"Data directory is not a directory: {root_dir}")
        return root_dir

    @staticmethod
    def _map_nyu40_label(label: torch.Tensor) -> torch.Tensor:
        label_np = label.numpy()
        mapped = np.full(label_np.shape, IGNORE_INDEX, dtype=np.int64)
        for source_id, target_id in NYU40_TO_5.items():
            mapped[label_np == source_id] = target_id
        return torch.from_numpy(mapped).long()

    @staticmethod
    def _validate_nyu5_label(label: torch.Tensor) -> torch.Tensor:
        valid = ((label >= 0) & (label < 5)) | (label == IGNORE_INDEX)
        if not bool(valid.all()):
            invalid_values = torch.unique(label[~valid]).tolist()
            raise ValueError(
                "NYU5 label contains values outside 0..4 and 255: "
                f"{invalid_values}"
            )
        return label.long()
