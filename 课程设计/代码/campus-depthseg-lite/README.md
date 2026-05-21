# Campus DepthSeg Lite

面向校园室内空间巡检的轻量 RGB-D 语义分割与空间占用分析系统。

本工程是课程设计第一轮代码框架与 CPU smoke test。当前不包含正式训练结果，不下载数据集，不下载大模型，也不会自动占用 GPU。

## Clean-Room 说明

- 本项目为独立实现，不复制、导入或依赖 `hansu650/Lun-Wen` 中的代码、类名、函数名、文件结构或实验配置。
- 本项目不复制 DFormer、DFormerv2、CMX、ESANet、TokenFusion 等开源论文项目源码。
- 模型仅参考 RGB-D 语义分割、深度边界先验和多尺度融合的通用思想，所有实现均为重新编写。

## 环境

```bash
pip install -r requirements.txt
```

## 快速验证

```bash
pytest -q
python scripts/smoke_forward.py --device cpu
python scripts/demo_inspection.py
```

`demo_inspection.py` 会生成：

```text
demo/results/demo_panel.png
```

## 数据格式

正式数据集使用 split 文件，每行三列：

```text
rgb_path depth_path label_path
```

路径可以是绝对路径，也可以是相对 split 文件所在目录的相对路径。若文件缺失，数据集会直接抛出清晰错误，不会自动生成假数据。

## 训练入口

```bash
python scripts/train.py \
  --train_split path/to/train.txt \
  --val_split path/to/val.txt \
  --accelerator gpu \
  --devices 1
```

CPU 快速调试可用：

```bash
python scripts/train.py \
  --train_split path/to/train.txt \
  --val_split path/to/val.txt \
  --accelerator cpu \
  --devices 1 \
  --fast_dev_run
```

## 当前范围

已包含：

- NYU40 到 5 类的基础标签映射
- RGB-D 数据读取与同步几何变换
- 轻量 RGB-D 语义分割模型
- Cross Entropy + Dice loss
- confusion matrix、pixel accuracy、mean accuracy、per-class IoU、mIoU
- Lightning 训练/验证/测试包装
- 空间占用风险分析
- 基于分割 mask 的 obstacle 连通域区域框
- synthetic-only smoke/demo/tests

暂未包含：

- 正式数据集下载与预处理脚本
- 正式训练实验与 checkpoint
- 小程序、报告、视频处理
- 更复杂的增强、蒸馏、量化或部署优化
