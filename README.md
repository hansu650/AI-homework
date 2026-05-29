# AI Homework

这个仓库用于保存课程实验与课程设计相关代码。当前重点项目是：

## 校园室内巡检的轻量 RGB-D 语义分割与空间占用分析系统

项目目录：[`课程设计/代码/campus-depthseg-lite`](课程设计/代码/campus-depthseg-lite)

该课程设计基于 NYUDepthV2 构建五类室内空间分割任务，比较 RGB-only、RGBD-concat、RGBD-boundary 和 RGBD-concat-boundary 四种模型变体，并提供训练曲线、测试集评估、预测可视化和自采集校园场景定性展示。

| Best Model | Test mIoU | Dataset | Model Size |
| --- | ---: | --- | ---: |
| RGBD-concat | **0.4683** | NYUDepthV2, 1449 samples | about 1.51M params |

<p align="center">
  <a href="课程设计/代码/campus-depthseg-lite">
    <strong>进入课程设计项目</strong>
  </a>
</p>

<p align="center">
  <img src="课程设计/代码/campus-depthseg-lite/web_demo/assets/architecture.png" alt="CampusDepthSegLite architecture" width="760">
</p>

## 静态展示页

课程设计项目提供了一个轻量静态 Web 展示页，只用于展示模型结构、实验结果和自采集校园场景定性结果，不进行在线推理，不加载 checkpoint，也不需要 GPU 或服务器。

入口：[`课程设计/代码/campus-depthseg-lite/web_demo/index.html`](课程设计/代码/campus-depthseg-lite/web_demo/index.html)

## 边界说明

- 数据集、训练输出、checkpoint、报告生成图片等本地大文件不纳入 Git 管理。
- 自采集校园图像没有像素级 GT，也没有真实 depth，仅用于定性展示。
- risk_score 是启发式提示，不代表真实安全决策标准。
- 本仓库不包含在线部署系统或实时推理服务。
