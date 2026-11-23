#!/usr/bin/env python3
"""
把本地 YOLO 数据集的图片 + 同名 .txt 标签，转换成 X-AnyLabeling 原生 JSON 标注，
直接放在图片同目录（与图片同名）。这样在 X-AnyLabeling 里「打开目录」后，
每一张图片都会自动显示标注框，不需要再用「上传 / 导入」功能。

原理：X-AnyLabeling 默认按「与图片同名的 .json」自动加载标注
（官方文档 1.4 保存标签数据：标签文件默认保存在当前图片目录下，格式为 *.json；
未设置输出目录时从图片目录查找同名 JSON）。

输出：flat/<split>/<img>.json  （与 flat/<split>/<img>.jpg 同名同目录）

依赖：pip install pillow

用法：
    python scripts/generate_xanylabeling_json.py --root "D:/检测"
"""

import argparse
import json
from pathlib import Path

from PIL import Image

DATASETS = ["electric-bicycle-entering-elevator", "lift-Ebike"]
VERSION = "3.2.4"


def read_classes(ds_dir):
    p = ds_dir / "classes.txt"
    if p.exists():
        names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if names:
            return names
    raise RuntimeError(f"找不到 classes.txt：{ds_dir}")


def parse_yolo(txt_path):
    boxes = []
    if not txt_path.is_file():
        return boxes
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        boxes.append((cid, xc, yc, bw, bh))
    return boxes


def make_shape(cid, xc, yc, bw, bh, img_w, img_h, names):
    """YOLO 归一化 xywh -> X-AnyLabeling 绝对像素 xyxy 矩形。"""
    cx, cy = xc * img_w, yc * img_h
    half_w, half_h = bw * img_w / 2, bh * img_h / 2
    x1 = max(0, round(cx - half_w, 2))
    y1 = max(0, round(cy - half_h, 2))
    x2 = min(img_w, round(cx + half_w, 2))
    y2 = min(img_h, round(cy + half_h, 2))
    label = names[cid] if 0 <= cid < len(names) else f"class{cid}"
    return {
        "label": label,
        "score": None,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": None,
        "difficult": False,
        "shape_type": "rectangle",
        "flags": None,
        "attributes": {},
    }


def convert_dataset(ds_dir, names):
    total = 0
    for split in ("train", "val", "test"):
        img_dir = ds_dir / "flat" / split
        if not img_dir.is_dir():
            continue
        n = 0
        for img_path in sorted(img_dir.glob("*.jpg")):
            txt = img_dir / (img_path.stem + ".txt")
            boxes = parse_yolo(txt)
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception as e:
                print(f"  跳过 {img_path.name}: 无法读取尺寸 {e}")
                continue
            shapes = [make_shape(*b, img_w, img_h, names) for b in boxes]
            data = {
                "version": VERSION,
                "flags": {},
                "tags": [],
                "shapes": shapes,
                "description": None,
                "checked": False,
                "chat_history": [],
                "vqaData": {},
                "imagePath": img_path.name,
                "imageData": None,
                "imageHeight": img_h,
                "imageWidth": img_w,
            }
            json_path = img_dir / (img_path.stem + ".json")
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            n += 1
        print(f"  {ds_dir.name}/{split}: 生成 {n} 个 JSON")
        total += n
    return total


def main():
    p = argparse.ArgumentParser(description="为 YOLO 数据集生成 X-AnyLabeling 原生 JSON 标注")
    p.add_argument("--root", default="D:/检测", help="数据集所在根目录")
    args = p.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"找不到目录：{root}")

    grand = 0
    for dirname in DATASETS:
        ds_dir = root / dirname
        if not ds_dir.is_dir():
            print(f"跳过：找不到 {ds_dir}")
            continue
        names = read_classes(ds_dir)
        print(f"=== {dirname}（{len(names)} 类）")
        grand += convert_dataset(ds_dir, names)

    print(f"\n共生成 {grand} 个 X-AnyLabeling JSON 标注文件。")
    print("现在到 X-AnyLabeling：文件 -> 打开目录，重新选 flat/train（或 val/test），")
    print("每张图片应自动显示标注框。若已打开该目录，请先切到别的目录再切回来，或重启软件。")


if __name__ == "__main__":
    main()
