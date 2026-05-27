# Campus DepthSeg Lite

Lightweight RGB-D semantic segmentation and occupancy analysis for campus indoor inspection.

This repository is an independent course-design implementation. It does not copy, import, or depend on private research code, unreleased project code, or external research-project source code. Data processing, model definition, training, evaluation, and visualization modules are written specifically for this course project.

## Scope

Current scope:

- NYUDepthV2 MAT inspection and local export to five classes.
- RGB-D dataset loading and synchronized transforms.
- Lightweight segmentation model with four experiment variants.
- Lightning training wrapper, CSV logs, best-checkpoint path, metric logging.
- Curve plotting, test-set evaluation, and prediction panels.
- Qualitative self-collected campus RGB demo panels.
- CPU smoke tests and synthetic demos.

Not included:

- Dataset downloads.
- Large pretrained model downloads.
- Training outputs, checkpoints, raw data, and generated report assets are not tracked by Git.
- Quantitative experiment results are summarized in `EXPERIMENT_RESULTS.md`.
- Mobile app, video processing, or report text.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Checks

```bash
python -m compileall src scripts tests
pytest -q
python scripts/smoke_forward.py --device cpu
python scripts/demo_inspection.py
```

## Data Layout

Prepared NYU5 data should live inside the project, but it is ignored by Git:

```text
data/nyu5/
  images/
  depths/
  labels/
  splits/
    train.txt
    val.txt
    test.txt
```

Each split line uses paths relative to `data/nyu5/`:

```text
images/000001.png depths/000001.png labels/000001.png
```

## Model Variants

| Experiment | Input | Depth Fusion | Purpose |
| --- | --- | --- | --- |
| RGB-only | RGB | none | visual baseline |
| RGBD-concat | RGB + depth | direct input concat | test direct depth input |
| RGBD-boundary | RGB + depth | Sobel depth boundary residual fusion | main method |
| RGBD-concat-boundary | RGB + depth | input concat + Sobel boundary residual fusion | test complementarity |

Valid `--variant` values:

```text
rgb
rgbd_concat
rgbd_boundary
rgbd_concat_boundary
```

## Experiment Commands

GPU fast development run:

```bash
python scripts/train.py --data_dir data/nyu5 --variant rgbd_boundary --experiment_name sanity_gpu --accelerator gpu --devices 1 --batch_size 2 --fast_dev_run
```

RGB baseline:

```bash
python scripts/train.py --data_dir data/nyu5 --variant rgb --experiment_name exp01_rgb --accelerator gpu --devices 1 --batch_size 4 --max_epochs 20
```

RGBD concat:

```bash
python scripts/train.py --data_dir data/nyu5 --variant rgbd_concat --experiment_name exp02_rgbd_concat --accelerator gpu --devices 1 --batch_size 4 --max_epochs 20
```

RGBD boundary main method:

```bash
python scripts/train.py --data_dir data/nyu5 --variant rgbd_boundary --experiment_name exp03_rgbd_boundary --accelerator gpu --devices 1 --batch_size 4 --max_epochs 20
```

RGBD concat + boundary complementarity:

```bash
python scripts/train.py --data_dir data/nyu5 --variant rgbd_concat_boundary --experiment_name exp04_rgbd_concat_boundary_e20 --accelerator gpu --devices 1 --batch_size 4 --max_epochs 20
```

Plot training curves:

```bash
python scripts/plot_training_curves.py --run_dir outputs/runs/exp03_rgbd_boundary
```

Evaluate the test set:

```bash
python scripts/evaluate.py --data_dir data/nyu5 --split_file data/nyu5/splits/test.txt --checkpoint outputs/runs/exp03_rgbd_boundary/checkpoints/best.ckpt --variant rgbd_boundary --batch_size 4 --accelerator gpu --devices 1
```

Prediction visualization:

```bash
python scripts/predict_folder.py --data_dir data/nyu5 --split_file data/nyu5/splits/test.txt --checkpoint outputs/runs/exp03_rgbd_boundary/checkpoints/best.ckpt --variant rgbd_boundary --num_samples 8 --out_dir outputs/runs/exp03_rgbd_boundary/predictions
```

Campus RGB qualitative demo:

```bash
python scripts/predict_campus_demo.py --rgb_dir data/campus_demo/rgb --checkpoint outputs/runs/exp01_rgb_e20/checkpoints/best.ckpt --variant rgb --out_dir outputs/report_assets/campus_demo --num_samples 8
```

The campus demo images have no pixel-level ground truth labels and no real depth.
They are used only for qualitative visualization with the RGB-only checkpoint.
They are not used for training, mIoU, pixel accuracy, mean accuracy, or other
quantitative evaluation.

## Outputs

Training outputs are written to:

```text
outputs/runs/{experiment_name}/
  metrics.csv
  checkpoints/best.ckpt
```

Generated data, logs, checkpoints, figures, and local MAT files are ignored by Git.
