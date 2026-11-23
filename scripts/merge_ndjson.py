#!/usr/bin/env python3
"""
把两个 Ultralytics Platform 导出的 NDJSON 数据集合并为一个 NDJSON，
用于训练统一类别的 YOLO11n 电梯内电动车检测模型。

数据集原始类别：
  electric-bicycle-entering-elevator (ds1):
      0 Baby-carriage / 1 Electric-bicycle / 2 bicycle / 3 person / 4 wheelchair
  lift-ebike (ds2):
      0 motorcycle （数据集描述为“电梯电动车”，实际标注为摩托车/电动车）

默认合并策略（5 类）：
  ds1 保持原类别编号
  ds2 的 0 motorcycle -> 统一类别 1 Electric-bicycle
  => 统一类别表: 0 Baby-carriage, 1 Electric-bicycle, 2 bicycle, 3 person, 4 wheelchair

如需保留 6 类（把 motorcycle 单独作为第 5 类），加参数 --keep-motorcycle
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DS1_NAMES = {0: "Baby-carriage", 1: "Electric-bicycle", 2: "bicycle", 3: "person", 4: "wheelchair"}
DS2_NAMES = {0: "motorcycle"}
UNIFIED_NAMES_5 = DS1_NAMES
UNIFIED_NAMES_6 = {**DS1_NAMES, 5: "motorcycle"}
BOX_KEY = "boxes"  # NDJSON 中检测框字段名，官方格式为 [class_id, xc, yc, w, h]


def read_ndjson(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    if not records or ("class_names" not in records[0] and records[0].get("type") != "dataset"):
        raise ValueError(f"{path} 第一行不是 dataset 记录")
    return records[0], records[1:]


def remap_records(image_records, class_map, n_classes, source_tag):
    """重写每张图的 boxes 类别编号；返回统计信息。"""
    out = []
    split_counter = Counter()
    class_counter = Counter()
    skipped = 0
    for rec in image_records:
        if rec.get("type", "image") != "image" or "file" not in rec:
            continue
        split = rec.get("split", "train")
        if split not in {"train", "val", "test"}:
            split = "train"
        split_counter[split] += 1
        rec = dict(rec)
        ann = dict(rec.get("annotations") or {})
        boxes = []
        for box in ann.get(BOX_KEY, []):
            old_id = int(box[0])
            if old_id not in class_map:
                skipped += 1
                continue
            new_id = class_map[old_id]
            if new_id >= n_classes:
                raise ValueError(f"{source_tag}: 映射后类别 {new_id} 超出类别数 {n_classes}")
            boxes.append([new_id, *box[1:]])
            class_counter[new_id] += 1
        if ann.get(BOX_KEY):
            ann[BOX_KEY] = boxes
            rec["annotations"] = ann
        out.append(rec)
    print(f"  {source_tag}: {len(out)} 张图, 划分={dict(split_counter)}, "
          f"重映射后标注={dict(class_counter)}, 丢弃非法框={skipped}")
    return out


def main():
    parser = argparse.ArgumentParser(description="合并两个 Ultralytics NDJSON 数据集")
    parser.add_argument("--ds1", default="data/electric-bicycle-entering-elevator.ndjson")
    parser.add_argument("--ds2", default="data/lift-ebike.ndjson")
    parser.add_argument("--out", default="data/merged_elevator_ebike.ndjson")
    parser.add_argument("--keep-motorcycle", action="store_true",
                        help="保留 motorcycle 为独立的第 5 类（默认映射到 Electric-bicycle）")
    args = parser.parse_args()

    names = UNIFIED_NAMES_6 if args.keep_motorcycle else UNIFIED_NAMES_5
    n_classes = len(names)
    ds2_map = {0: 5} if args.keep_motorcycle else {0: 1}

    print("===== 读取数据集 =====")
    ds1_meta, ds1_imgs = read_ndjson(args.ds1)
    ds2_meta, ds2_imgs = read_ndjson(args.ds2)
    print("ds1 原始类别:", ds1_meta.get("class_names"))
    print("ds2 原始类别:", ds2_meta.get("class_names"))
    print("统一类别    :", names)

    # 校验原始类别，防止平台改版后静默出错
    if {int(k): v for k, v in ds1_meta.get("class_names", {}).items()} != DS1_NAMES:
        print("警告: ds1 类别与脚本预期不一致，请检查是否需要修改 DS1_NAMES")
    if {int(k): v for k, v in ds2_meta.get("class_names", {}).items()} != DS2_NAMES:
        print("警告: ds2 类别与脚本预期不一致，请检查是否需要修改 DS2_NAMES")

    print("===== 类别重映射 =====")
    img1 = remap_records(ds1_imgs, {i: i for i in DS1_NAMES}, n_classes, "ds1")
    img2 = remap_records(ds2_imgs, ds2_map, n_classes, "ds2")

    merged_meta = dict(ds1_meta)
    merged_meta.update({
        "type": "dataset",
        "task": "detect",
        "name": "elevator-ebike-merged",
        "description": "电梯内电动车检测：合并 elevator-entering + lift-ebike（类别已统一）",
        "class_names": {str(k): v for k, v in names.items()},
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(merged_meta, ensure_ascii=False) + "\n")
        for rec in img1 + img2:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"===== 完成 =====\n已生成: {out_path.resolve()}")
    print(f"总计 {len(img1) + len(img2)} 张图片, {n_classes} 个类别")
    print("训练时直接使用: data=" + str(out_path))


if __name__ == "__main__":
    main()
