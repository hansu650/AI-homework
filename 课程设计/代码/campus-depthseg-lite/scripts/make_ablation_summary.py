"""Create ablation result tables from completed test_metrics.json files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES

EXPERIMENTS = [
    {
        "run": "exp01_rgb_e20",
        "method": "RGB-only",
        "input": "RGB",
        "fusion": "none",
    },
    {
        "run": "exp02_rgbd_concat_e20",
        "method": "RGBD-concat",
        "input": "RGB + Depth",
        "fusion": "input concat",
    },
    {
        "run": "exp03_rgbd_boundary_e20",
        "method": "RGBD-boundary",
        "input": "RGB + Depth",
        "fusion": "depth boundary residual fusion",
    },
    {
        "run": "exp04_rgbd_concat_boundary_e20",
        "method": "RGBD-concat-boundary",
        "input": "RGB + Depth",
        "fusion": "input concat + boundary residual fusion",
    },
]


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "report_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows = []
    per_class_rows = []
    for experiment in EXPERIMENTS:
        metrics = _read_metrics(experiment["run"])
        ablation_rows.append(
            {
                "method": experiment["method"],
                "input": experiment["input"],
                "fusion": experiment["fusion"],
                "test_mIoU": metrics["test_mIoU"],
                "pixel_acc": metrics["test_pixel_acc"],
                "mean_acc": metrics["test_mean_acc"],
                "test_loss": metrics["test_loss"],
            }
        )
        per_class = {
            "method": experiment["method"],
            "input": experiment["input"],
            "fusion": experiment["fusion"],
        }
        for class_name in CLASS_NAMES:
            per_class[class_name] = metrics["per_class_iou"][class_name]
        per_class_rows.append(per_class)

    ablation_fields = [
        "method",
        "input",
        "fusion",
        "test_mIoU",
        "pixel_acc",
        "mean_acc",
        "test_loss",
    ]
    per_class_fields = ["method", "input", "fusion", *CLASS_NAMES]

    _write_csv(out_dir / "ablation_results.csv", ablation_fields, ablation_rows)
    _write_markdown(out_dir / "ablation_results.md", ablation_fields, ablation_rows)
    _write_csv(out_dir / "per_class_iou.csv", per_class_fields, per_class_rows)
    _write_markdown(out_dir / "per_class_iou.md", per_class_fields, per_class_rows)
    print(f"ablation tables saved under: {out_dir}")


def _read_metrics(run_name: str) -> dict[str, object]:
    metrics_path = PROJECT_ROOT / "outputs" / "runs" / run_name / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"test_metrics.json not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


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
        lines.append("| " + " | ".join(_format_value(row[name]) for name in field_names) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
