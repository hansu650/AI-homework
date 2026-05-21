"""Audit NYUDepthV2 raw label names mapped to five course classes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES
from src.utils.label_mapping import build_name_to_five_class_mapping
from src.utils.nyu_io import find_nyu_labeled_mat, read_nyu_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit NYU raw label mapping.")
    parser.add_argument("--data_root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mat_path = find_nyu_labeled_mat(Path(args.data_root))
    names = read_nyu_names(mat_path)
    id_to_five = build_name_to_five_class_mapping(names)

    lines = [
        f"MAT file: {mat_path}",
        "Mapping rule note: label id 1 corresponds to names[0]; label id 0 maps to other.",
        "Change note: singular 'book' is included as obstacle; floor matches exact 'floor' only.",
        "",
        "raw_label_id\traw_class_name\tmapped_5class_id\tmapped_5class_name",
    ]
    for raw_id in range(0, len(names) + 1):
        raw_name = "<unlabeled_or_zero>" if raw_id == 0 else names[raw_id - 1]
        mapped_id = id_to_five[raw_id]
        lines.append(
            f"{raw_id}\t{raw_name}\t{mapped_id}\t{CLASS_NAMES[mapped_id]}"
        )

    output_path = PROJECT_ROOT / "outputs" / "figures" / "label_mapping_report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"label mapping report saved: {output_path}")
    print(f"raw classes: {len(names)}")


if __name__ == "__main__":
    main()
