把你自己训练的模型导出成 ONNX 后，命名为 best.onnx 放在这个文件夹：
    docs/assets/models/best.onnx

导出步骤（Ultralytics Platform）：
1. 打开你的模型页面（Models → yolo11n_ds1_elevator 或 yolo11n_ds2_liftebike）
2. 点 Export 标签页
3. 格式选 ONNX，imgsz 选 640
4. Start Export，完成后下载 best.onnx
5. 重命名为 best.onnx 放到本目录，重新推送到 GitHub 即可生效

注意：
- 如果用数据集1的模型（5类），detect.js 里的 CLASS_NAMES 保持默认即可
- 如果用数据集2的模型（1类 motorcycle），把 CLASS_NAMES 改成 ["motorcycle"]
- ONNX 约 10MB，GitHub 仓库完全放得下
- 浏览器用的是 wasm CPU 推理，第一次加载需要十几秒
