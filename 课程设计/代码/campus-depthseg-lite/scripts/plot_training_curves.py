"""Plot training curves from a CSVLogger metrics.csv file."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves.")
    parser.add_argument("--run_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    rows = _read_metrics(metrics_path)
    _plot_metric(rows, ["train_loss_epoch", "train_loss"], run_dir / "training_loss_curve.png")
    _plot_metric(rows, ["val_mIoU"], run_dir / "validation_miou_curve.png")
    _plot_metric(rows, ["val_pixel_acc"], run_dir / "validation_pixel_acc_curve.png")
    print(f"curves saved under: {run_dir}")


def _read_metrics(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_metric(rows: list[dict[str, str]], metric_names: list[str], output_path: Path) -> None:
    metric_name = _first_existing_metric(rows, metric_names)
    points = []
    for row in rows:
        value = row.get(metric_name, "")
        if value == "":
            continue
        x_value = row.get("epoch") or row.get("step")
        if x_value == "":
            raise ValueError(f"Row for {metric_name} has no epoch or step value")
        points.append((float(x_value), float(value)))

    if len(points) == 0:
        raise ValueError(f"No values found for metric {metric_name}")

    xs, ys = zip(*points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(xs, ys, marker="o")
    axis.set_xlabel("epoch")
    axis.set_ylabel(metric_name)
    axis.set_title(metric_name)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _first_existing_metric(rows: list[dict[str, str]], metric_names: list[str]) -> str:
    if len(rows) == 0:
        raise ValueError("metrics.csv contains no rows")
    field_names = rows[0].keys()
    for name in metric_names:
        if name in field_names:
            return name
    raise ValueError(f"None of these metrics were found: {metric_names}")


if __name__ == "__main__":
    main()
