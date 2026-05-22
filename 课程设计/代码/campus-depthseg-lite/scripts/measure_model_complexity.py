"""Measure parameter counts for all model variants."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.campus_depthseg_lite import CampusDepthSegLite

VARIANTS = ["rgb", "rgbd_concat", "rgbd_boundary", "rgbd_concat_boundary"]


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "report_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        model = CampusDepthSegLite(variant=variant)
        params = sum(parameter.numel() for parameter in model.parameters())
        rows.append(
            {
                "variant": variant,
                "params": params,
                "params_million": params / 1_000_000.0,
            }
        )

    fields = ["variant", "params", "params_million"]
    _write_csv(out_dir / "model_complexity.csv", fields, rows)
    _write_markdown(out_dir / "model_complexity.md", fields, rows)
    print(f"model complexity tables saved under: {out_dir}")


def _write_csv(
    output_path: Path,
    field_names: list[str],
    rows: list[dict[str, object]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    output_path: Path,
    field_names: list[str],
    rows: list[dict[str, object]],
) -> None:
    lines = [
        "| " + " | ".join(field_names) + " |",
        "| " + " | ".join("---" for _ in field_names) + " |",
    ]
    for row in rows:
        values = []
        for name in field_names:
            value = row[name]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
