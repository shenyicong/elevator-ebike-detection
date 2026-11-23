本目录存放浏览器在线检测用的 ONNX 模型（由自己训练的 .pt 权重导出）。

ds1_elevator.onnx  - 数据集1模型，5类（Baby-carriage/Electric-bicycle/bicycle/person/wheelchair）
ds2_liftebike.onnx - 数据集2模型，1类（motorcycle）

导出来源：
  D:/edge.download/yolo11nds1elevator.pt  -> ds1_elevator.onnx
  D:/edge.download/yolo11nds2liftebike.pt -> ds2_liftebike.onnx
导出命令：YOLO(pt).export(format='onnx', imgsz=640, opset=17, simplify=True)
验证：已用 onnxruntime(Python) 与 onnxruntime-web(Node) 双向验证，输出 shape [1,9,8400]/[1,5,8400]，检测结果正确。

如需更新模型：用新权重重新导出为 ONNX 覆盖对应文件即可，类别顺序要同步修改 docs/assets/detect.js 里的 MODELS[key].names。
浏览器用 wasm CPU 推理，首次加载约十几秒，模型会被浏览器缓存。
