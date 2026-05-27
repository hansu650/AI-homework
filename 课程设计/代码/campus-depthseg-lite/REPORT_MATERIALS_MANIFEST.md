# Report Materials Manifest

This file is a planning checklist for the course design report. It is not the report body.

## Project Title

面向校园室内空间巡检的轻量 RGB-D 语义分割与空间占用分析系统

## Report Template Basis

- Organize the report according to the course design template: abstract, introduction, related theory and techniques, dataset and preprocessing, model design and experimental setup, experimental results and analysis, conclusion and outlook, and appendices.
- Later formatting should follow the teacher's template requirements: body text 小四, page margins 2 cm, first-level headings 黑体小三, second-level headings 黑体四号, and 1.5 line spacing.

## Dataset Information

- Dataset: NYUDepthV2
- Total samples: 1449
- Train samples: 1014
- Validation samples: 217
- Test samples: 218
- Five labels: `other`, `floor`, `wall`, `obstacle`, `door_window`
- Input size: 240 x 320

## Experiment Results

| Method | mIoU | Pixel Acc | Mean Acc | Test Loss |
| --- | ---: | ---: | ---: | ---: |
| RGB-only | 0.4226 | 0.5964 | 0.5881 | 1.5588 |
| RGBD-concat | 0.4683 | 0.6318 | 0.6294 | 1.4205 |
| RGBD-boundary | 0.4206 | 0.5974 | 0.5925 | 1.5317 |
| RGBD-concat-boundary | 0.4605 | 0.6271 | 0.6191 | 1.4378 |

## Core Conclusions

- The final system should use RGBD-concat because it has the best overall test mIoU, 0.4683.
- RGBD-concat improves mIoU by 4.57 percentage points over RGB-only, indicating that depth information is useful.
- RGBD-concat-boundary has the best obstacle IoU, indicating that boundary cues can help obstacle-focused inspection.
- Boundary fusion does not exceed concat in overall mIoU, so it must not be described as the overall best variant.
- The self-collected campus demo is qualitative only and must not be described as quantitative testing.

## Available Report Assets

| Status | Path |
| --- | --- |
| OK | `outputs/figures/nyu5_class_distribution.png` |
| OK | `outputs/figures/nyu5_gallery_train.png` |
| OK | `outputs/figures/nyu5_gallery_val.png` |
| OK | `outputs/figures/nyu5_gallery_test.png` |
| OK | `outputs/report_assets/ablation_results.md` |
| OK | `outputs/report_assets/per_class_iou.md` |
| OK | `outputs/report_assets/model_complexity.md` |
| OK | `outputs/report_assets/method_comparison/method_comparison_grid.png` |
| OK | `outputs/report_assets/confusion_matrix/confusion_matrix_normalized.png` |
| OK | `outputs/report_assets/error_cases/best_cases.png` |
| OK | `outputs/report_assets/error_cases/worst_cases.png` |
| OK | `outputs/report_assets/campus_demo/campus_demo_gallery.png` |
| OK | `outputs/report_assets/campus_demo/campus_demo_summary.md` |
| OK | `outputs/runs/exp02_rgbd_concat_e20/training_loss_curve.png` |
| OK | `outputs/runs/exp02_rgbd_concat_e20/validation_miou_curve.png` |
| OK | `outputs/runs/exp02_rgbd_concat_e20/validation_pixel_acc_curve.png` |

## Suggested Report Sections

- 摘要
- 1 引言
- 2 相关理论与技术
- 3 问题定义与数据集
- 4 模型设计与系统实现
- 5 实验结果与分析
- 6 总结与展望
- 参考文献
- 附录A 核心代码说明
- 附录B 项目运行命令

## Claims To Avoid

- Do not claim quantitative accuracy on self-collected campus images.
- Do not claim that self-collected campus images have ground truth labels.
- Do not claim that self-collected campus images have real depth.
- Do not claim that the campus demo uses the RGB-D best model.
- Do not claim that boundary fusion is the overall best model.
- Do not describe the RGB-only campus demo as an RGB-D result.
- Do not claim the system can be directly used for real safety decisions.
- Do not claim that the model accurately identifies all obstacles.
