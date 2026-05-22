# Experiment Results

This file records the completed local training results. Large generated artifacts
such as checkpoints, curves, prediction panels, and exported data remain ignored
by Git under `outputs/` and `data/`.

## Dataset

- Dataset: local NYUDepthV2 exported to NYU5.
- Classes: `other`, `floor`, `wall`, `obstacle`, `door_window`.
- Split sizes:
  - train: 1014
  - val: 217
  - test: 218
- Input size: 240 x 320

## Training Setup

- Epochs: 20 for each ablation.
- Batch size: 4
- Optimizer: AdamW
- Learning rate: 1e-4
- Loss: CrossEntropyLoss + DiceLoss
- Hardware: single GPU
- DataLoader workers: 0

## Validation Best Checkpoints

| Experiment | Variant | Best val mIoU | Checkpoint |
| --- | --- | ---: | --- |
| `exp01_rgb_e20` | `rgb` | 0.4082 | `outputs/runs/exp01_rgb_e20/checkpoints/best.ckpt` |
| `exp02_rgbd_concat_e20` | `rgbd_concat` | 0.4576 | `outputs/runs/exp02_rgbd_concat_e20/checkpoints/best.ckpt` |
| `exp03_rgbd_boundary_e20` | `rgbd_boundary` | 0.4049 | `outputs/runs/exp03_rgbd_boundary_e20/checkpoints/best.ckpt` |
| `exp04_rgbd_concat_boundary_e20` | `rgbd_concat_boundary` | 0.4524 | `outputs/runs/exp04_rgbd_concat_boundary_e20/checkpoints/best.ckpt` |

## Test Metrics

| Method | Input | Fusion | mIoU | Pixel Acc | Mean Acc | Test Loss |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| RGB-only | RGB | none | 0.4226 | 0.5964 | 0.5881 | 1.5588 |
| RGBD-concat | RGB + Depth | input concat | 0.4683 | 0.6318 | 0.6294 | 1.4205 |
| RGBD-boundary | RGB + Depth | depth boundary residual fusion | 0.4206 | 0.5974 | 0.5925 | 1.5317 |
| RGBD-concat-boundary | RGB + Depth | input concat + depth boundary residual fusion | 0.4605 | 0.6271 | 0.6191 | 1.4378 |

## Per-Class IoU

| Method | other | floor | wall | obstacle | door_window |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB-only | 0.3549 | 0.4731 | 0.5495 | 0.4239 | 0.3114 |
| RGBD-concat | 0.3857 | 0.5751 | 0.5647 | 0.4722 | 0.3437 |
| RGBD-boundary | 0.3476 | 0.4629 | 0.5443 | 0.4276 | 0.3205 |
| RGBD-concat-boundary | 0.3636 | 0.5530 | 0.5595 | 0.4899 | 0.3366 |

## Local Artifacts

Training curves:

- `outputs/runs/exp01_rgb_e20/training_loss_curve.png`
- `outputs/runs/exp01_rgb_e20/validation_miou_curve.png`
- `outputs/runs/exp01_rgb_e20/validation_pixel_acc_curve.png`
- `outputs/runs/exp02_rgbd_concat_e20/training_loss_curve.png`
- `outputs/runs/exp02_rgbd_concat_e20/validation_miou_curve.png`
- `outputs/runs/exp02_rgbd_concat_e20/validation_pixel_acc_curve.png`
- `outputs/runs/exp03_rgbd_boundary_e20/training_loss_curve.png`
- `outputs/runs/exp03_rgbd_boundary_e20/validation_miou_curve.png`
- `outputs/runs/exp03_rgbd_boundary_e20/validation_pixel_acc_curve.png`
- `outputs/runs/exp04_rgbd_concat_boundary_e20/training_loss_curve.png`
- `outputs/runs/exp04_rgbd_concat_boundary_e20/validation_miou_curve.png`
- `outputs/runs/exp04_rgbd_concat_boundary_e20/validation_pixel_acc_curve.png`

Prediction panels for the boundary model:

- `outputs/runs/exp03_rgbd_boundary_e20/predictions/prediction_001.png`
- `outputs/runs/exp03_rgbd_boundary_e20/predictions/prediction_002.png`
- `outputs/runs/exp03_rgbd_boundary_e20/predictions/prediction_003.png`
- `outputs/runs/exp03_rgbd_boundary_e20/predictions/prediction_004.png`
- `outputs/runs/exp03_rgbd_boundary_e20/predictions/prediction_005.png`
- `outputs/runs/exp03_rgbd_boundary_e20/predictions/prediction_006.png`
- `outputs/runs/exp03_rgbd_boundary_e20/predictions/prediction_007.png`
- `outputs/runs/exp03_rgbd_boundary_e20/predictions/prediction_008.png`

## Observation

Direct RGB-D concatenation is the strongest overall variant in this run. It
improves over RGB-only on mIoU, pixel accuracy, mean accuracy, and every
per-class IoU. Adding depth-boundary residual fusion on top of RGB-D
concatenation improves obstacle IoU over plain concatenation, but slightly
reduces overall mIoU. The boundary prior should therefore be described as useful
for obstacle-focused behavior but not as the best aggregate segmentation model
under the current lightweight implementation.
