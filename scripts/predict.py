#!/usr/bin/env python3
"""
用训练好的 best.pt 做推理 / 导出，供论文截图与部署测试。

示例：
  python scripts/predict.py --weights runs/detect/yolo11n_merged/weights/best.pt \
      --source data/test_images --device 0
  python scripts/predict.py --weights runs/detect/yolo11n_merged/weights/best.pt --export
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="YOLO11n 电梯内电动车检测推理/导出")
    p.add_argument("--weights", required=True, help="训练得到的 best.pt 路径")
    p.add_argument("--source", default="data/test_images", help="图片/文件夹/视频路径，或 0 表示摄像头")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--export", action="store_true", help="只导出 ONNX，不做推理")
    p.add_argument("--format", default="onnx", help="导出格式，如 onnx / openvino / engine")
    args = p.parse_args()

    model = YOLO(args.weights)
    if args.export:
        path = model.export(format=args.format, device=args.device)
        print(f"导出完成: {path}")
        return

    source = args.source
    if source != "0" and not Path(source).exists():
        raise SystemExit(f"找不到推理源: {source}")

    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=True,
        project="runs/predict",
        name="elevator_ebike",
        exist_ok=True,
    )
    print(f"推理完成，共 {len(results)} 张/帧，结果保存在 runs/predict/elevator_ebike")


if __name__ == "__main__":
    main()
