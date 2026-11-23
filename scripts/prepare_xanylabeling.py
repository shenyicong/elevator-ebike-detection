#!/usr/bin/env python3
r"""
为 X-AnyLabeling 导入 YOLO 标注做准备。

X-AnyLabeling 打开图片文件夹时【不会自动显示】同名的 YOLO .txt 标签，
必须用菜单「上传」-> 目标检测(YOLO) -> 选 classes.txt + 标签文件目录 来导入。
而且官方建议图片和 txt 不要混在同一个目录。

本脚本对每个数据集：
  1) 根据 data.yaml 生成 classes.txt（类别名按类别 id 顺序）
  2) 把所有 split 的 .txt 汇总复制到单独的 labels/ 目录（文件名全局唯一，不会冲突）
  3) 图片仍保留在 flat/train、flat/val 等原位置

用法：
  python scripts/prepare_xanylabeling.py --root "D:\检测"
"""

import argparse
import re
from pathlib import Path


def parse_names(yaml_path: Path):
    names = {}
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            m = re.match(r"^\s+(\d+):\s*(.+?)\s*$", line)
            if m:
                names[int(m.group(1))] = m.group(2)
            elif line.strip():
                in_names = False
    return names


def prepare(ds_dir: Path):
    yaml_path = ds_dir / "data.yaml"
    if not yaml_path.is_file():
        print(f"跳过 {ds_dir}：没有 data.yaml")
        return
    names = parse_names(yaml_path)
    if not names:
        print(f"跳过 {ds_dir}：data.yaml 里没解析到类别")
        return

    classes_path = ds_dir / "classes.txt"
    classes_path.write_text("\n".join(names[i] for i in sorted(names)) + "\n", encoding="utf-8")

    label_dir = ds_dir / "labels"
    label_dir.mkdir(exist_ok=True)
    count = 0
    for src in ds_dir.rglob("*.txt"):
        if src.parent == label_dir or src.name == "classes.txt" or src.name == "download_failed.txt":
            continue
        dst = label_dir / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        count += 1

    # X-AnyLabeling 打开 flat/train 时，会去图片目录【同级】的 labels/ 下按 split 找标签。
    # 因此还要在 flat/labels/<split>/ 里放一份，才能让每一张图都自动显示框。
    mirrored = 0
    for split_dir in sorted((ds_dir / "flat").iterdir()):
        if not split_dir.is_dir() or split_dir.name == "labels":
            continue
        target = ds_dir / "flat" / "labels" / split_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for src in split_dir.glob("*.txt"):
            dst = target / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            mirrored += 1

    print(f"{ds_dir.name}: classes.txt ({len(names)} 类), labels/ 汇总 {count} 个 txt, flat/labels/ 镜像 {mirrored} 个")
    print(f"  classes.txt -> {classes_path}")
    print(f"  labels/     -> {label_dir}")
    print(f"  flat/labels/{'<split>'} -> 与 flat/train、flat/val 同级，供 X-AnyLabeling 自动匹配")


def main():
    p = argparse.ArgumentParser(description="为 X-AnyLabeling 准备 classes.txt 和 labels 目录")
    p.add_argument("--root", default=r"D:\检测", help="包含各数据集文件夹的根目录")
    args = p.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"找不到目录: {root}")
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        prepare(d)


if __name__ == "__main__":
    main()
