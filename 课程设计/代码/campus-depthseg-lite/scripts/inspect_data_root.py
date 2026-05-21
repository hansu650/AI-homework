"""Inspect the local NYUDepthV2 data root without loading large arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.nyu_io import find_nyu_labeled_mat, inspect_mat_file

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OTHER_EXTENSIONS = {".mat", ".npy", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect NYUDepthV2 data root.")
    parser.add_argument("--data_root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not data_root.is_dir():
        raise ValueError(f"Data root is not a directory: {data_root}")

    lines = build_report(data_root)
    report_path = PROJECT_ROOT / "outputs" / "figures" / "data_inspection_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    text = "\n".join(lines)
    print(text)
    print(f"\nReport saved: {report_path}")


def build_report(data_root: Path) -> list[str]:
    lines: list[str] = []
    lines.append(f"Data root: {data_root}")
    lines.append("")
    lines.append("Top-level entries:")
    for entry in sorted(data_root.iterdir(), key=lambda item: item.name.lower()):
        kind = "dir" if entry.is_dir() else "file"
        size = entry.stat().st_size if entry.is_file() else "-"
        extension = entry.suffix if entry.is_file() else "-"
        lines.append(f"- {entry.name}\t{kind}\tsize={size}\text={extension}")

    lines.append("")
    lines.append("Expected entries:")
    for name in [
        "NYUDepthv2",
        "NYUDepthv2_matdepth",
        "nyu_depth_v2_labeled",
        "nyu_depth_v2_labeled.mat",
    ]:
        lines.append(f"- {name}: {(data_root / name).exists()}")

    lines.append("")
    lines.append("Folder summaries:")
    for entry in sorted(data_root.iterdir(), key=lambda item: item.name.lower()):
        if entry.is_dir():
            lines.extend(_summarize_folder(entry))

    lines.append("")
    lines.append("MAT file inspection:")
    mat_path = find_nyu_labeled_mat(data_root)
    mat_info = inspect_mat_file(mat_path)
    lines.append(f"- path: {mat_info['path']}")
    if not mat_info["is_hdf5"]:
        lines.append(f"- h5py readable: False")
        lines.append(f"- h5py error: {mat_info['error']}")
        lines.append(
            "- note: This may be a pre-v7.3 MAT file. Use scipy.io.loadmat "
            "for a targeted format check in a later step; this script does "
            "not load large arrays."
        )
        return lines

    lines.append("- h5py readable: True")
    lines.append(f"- sample_count: {mat_info['sample_count']}")
    lines.append(f"- has_names: {mat_info['has_names']}")
    if not mat_info["has_names"]:
        lines.append("- note: names missing; raw label id mapping needs manual confirmation.")
    lines.append("- keys:")
    for key in mat_info["keys"]:
        lines.append(f"  - {key}")
    lines.append("- datasets:")
    for key, meta in mat_info["datasets"].items():
        lines.append(f"  - {key}: shape={meta['shape']}, dtype={meta['dtype']}")
    return lines


def _summarize_folder(folder: Path) -> list[str]:
    files = [path for path in folder.rglob("*") if path.is_file()]
    image_count = sum(1 for path in files if path.suffix.lower() in IMAGE_EXTENSIONS)
    other_counts = {
        extension: sum(1 for path in files if path.suffix.lower() == extension)
        for extension in sorted(OTHER_EXTENSIONS)
    }

    lines = [f"- {folder.name}:"]
    lines.append(f"  images(.png/.jpg/.jpeg): {image_count}")
    for extension, count in other_counts.items():
        lines.append(f"  {extension}: {count}")
    lines.append("  first 10 files:")
    for path in files[:10]:
        lines.append(f"    {path.relative_to(folder)}")
    return lines


if __name__ == "__main__":
    main()
