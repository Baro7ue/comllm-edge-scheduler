# comllm-edge-scheduler

基于开源论文 COMLLM（arXiv:2604.07148）的任务卸载调度器端侧部署：Jetson Orin Nano 8GB 做调度节点（llama.cpp/CUDA），RK3588 做环境节点（开发期用 Mac 代替），把决策延迟计入闭环评测。

> 开发中（2026 年 11 月）。结果表只放实测值，完成后补全。

## 目录

- sim/：仿真器和基线
- deploy/：Jetson 调度服务
- train/：Colab 训练 notebook
- eval/：评测与出图
- runs/：每次运行的结果，目录名为 UTC 时间加名字
- docs/：版本记录等文档
- evidence/：各阶段验收证据
