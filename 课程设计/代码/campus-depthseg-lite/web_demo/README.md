# 静态 Web 展示页

本目录是课程设计项目的轻量静态展示页，只用于展示报告结果，不是在线推理系统。

## 使用方式

在线预览：

[`点击查看静态展示页`](https://htmlpreview.github.io/?https://github.com/hansu650/AI-homework/blob/main/%E8%AF%BE%E7%A8%8B%E8%AE%BE%E8%AE%A1/%E4%BB%A3%E7%A0%81/campus-depthseg-lite/web_demo/index.html)

本地查看：

直接双击 `index.html` 即可在浏览器中查看。GitHub 的文件浏览页会把 HTML 当作源码显示，不会直接渲染页面。该页面不需要服务器、GPU、模型检查点或数据集。

## 展示内容

- 项目简介、五类标签和关键成果指标。
- CampusDepthSegLite 方法概览。
- 四组 RGB-D 融合策略消融实验结果。
- 训练过程曲线、预测对比图和混淆矩阵。
- 自采集校园场景样例与定性展示。
- 局限性和本地运行命令。

## 素材说明

`assets/` 中只保存少量报告使用图片，不包含原始数据集、训练输出目录或模型权重。若图片缺失，需要先从报告素材中复制对应图片后刷新页面。

当前使用图片：

- `assets/architecture.png`
- `assets/training_process.png`
- `assets/method_comparison.png`
- `assets/confusion_matrix.png`
- `assets/campus_rgb_gallery.png`
- `assets/campus_classroom.png`
- `assets/campus_corridor.png`
- `assets/campus_lab.png`
- `assets/campus_occluded.png`

## 边界说明

自采集校园图片没有像素级 GT，也没有真实 depth，仅用于定性展示，不参与训练和定量评价。页面中的 risk_score 是启发式提示，不代表真实安全决策标准。
