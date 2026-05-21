"""Prepare a five-class NYU5 dataset from a local NYUDepthV2 MAT file."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES
from src.utils.label_mapping import build_name_to_five_class_mapping, remap_label_by_id
from src.utils.nyu_io import (
    find_nyu_labeled_mat,
    inspect_mat_file,
    normalize_depth_to_uint16,
    read_nyu_mat_sample,
    read_nyu_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NYUDepthV2 to NYU5 PNG files.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="data/nyu5")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_ratios(args.train_ratio, args.val_ratio)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    mat_path = find_nyu_labeled_mat(data_root)
    mat_info = inspect_mat_file(mat_path)
    if not mat_info["is_hdf5"]:
        raise RuntimeError(
            "nyu_depth_v2_labeled is not readable by h5py. "
            "This preparation script only supports v7.3 HDF5 MAT files for now."
        )
    if not mat_info["has_names"]:
        raise KeyError(
            "Dataset 'names' missing; cannot build keyword-based label mapping "
            "without manual label-id confirmation."
        )

    names = read_nyu_names(mat_path)
    id_to_five = build_name_to_five_class_mapping(names)

    sample_count = int(mat_info["sample_count"])
    export_count = sample_count if args.max_samples is None else min(args.max_samples, sample_count)
    if export_count <= 0:
        raise ValueError("max_samples must allow at least one sample to be exported")

    _prepare_output_dirs(out_dir)
    class_pixel_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    records: list[str] = []
    for index in range(export_count):
        rgb, depth, raw_label = read_nyu_mat_sample(mat_path, index)
        label = remap_label_by_id(raw_label, id_to_five)
        class_pixel_counts += np.bincount(
            label.reshape(-1),
            minlength=256,
        )[: len(CLASS_NAMES)]

        stem = f"{index + 1:06d}"
        Image.fromarray(rgb).save(out_dir / "images" / f"{stem}.png")
        Image.fromarray(normalize_depth_to_uint16(depth)).save(
            out_dir / "depths" / f"{stem}.png"
        )
        Image.fromarray(label.astype(np.uint8)).save(out_dir / "labels" / f"{stem}.png")
        records.append(f"images/{stem}.png depths/{stem}.png labels/{stem}.png")

    split_records = _split_records(records, args.train_ratio, args.val_ratio, args.seed)
    for split_name, split_lines in split_records.items():
        (out_dir / "splits" / f"{split_name}.txt").write_text(
            "\n".join(split_lines) + "\n",
            encoding="utf-8",
        )

    print(f"MAT file: {mat_path}")
    print(f"Total samples in MAT: {sample_count}")
    print(f"Exported samples: {export_count}")
    print(f"Output directory: {out_dir}")
    for split_name, split_lines in split_records.items():
        print(f"{split_name}: {len(split_lines)}")
    print("Class pixel counts:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id} {class_name}: {int(class_pixel_counts[class_id])}")


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0.0 or val_ratio < 0.0:
        raise ValueError("train_ratio must be positive and val_ratio must be non-negative")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")


def _prepare_output_dirs(out_dir: Path) -> None:
    for name in ["images", "depths", "labels", "splits"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def _split_records(
    records: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)

    train_count = int(len(records) * train_ratio)
    val_count = int(len(records) * val_ratio)
    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count :]

    return {
        "train": [records[index] for index in train_indices],
        "val": [records[index] for index in val_indices],
        "test": [records[index] for index in test_indices],
    }


if __name__ == "__main__":
    main()
