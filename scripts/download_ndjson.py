#!/usr/bin/env python3
r"""
把从 Ultralytics Platform 网页下载的 .ndjson 数据集，真正下载成电脑上的
「图片 + YOLO .txt 标签」文件夹，效果类似你截图里的 D:\car（jpg 和 txt 成对出现）。

只依赖 Python 标准库，不需要额外 pip 安装。

示例（Windows PowerShell / CMD）：
  python scripts\download_ndjson.py ^
    --ndjson data\electric-bicycle-entering-elevator.ndjson ^
    --ndjson data\lift-ebike.ndjson ^
    --out "C:\Users\Q2730\Desktop\elevator_ebike_dataset"

合并后的数据集也支持：
  python scripts\merge_ndjson.py
  python scripts\download_ndjson.py --ndjson data\merged_elevator_ebike.ndjson ^
    --out "C:\Users\Q2730\Desktop\elevator_ebike_dataset"
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SPLITS = ("train", "val", "test")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "dataset"


def read_records(path: Path):
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    records = [json.loads(ln) for ln in lines]
    if not records or "class_names" not in records[0]:
        raise ValueError(f"{path} 不是有效的 Ultralytics NDJSON 文件")
    return records[0], records[1:]


def label_lines(record: dict) -> str:
    """从 NDJSON 的 annotations.boxes 生成 YOLO txt 内容。"""
    ann = record.get("annotations") or {}
    boxes = ann.get("boxes") or []
    out = []
    for box in boxes:
        if len(box) < 5:
            continue
        cls = int(box[0])
        coords = " ".join(str(v) for v in box[1:5])
        out.append(f"{cls} {coords}")
    return "\n".join(out) + ("\n" if out else "")


def download_file(url: str, dest: Path, retries: int = 3):
    if dest.exists() and dest.stat().st_size > 0:
        return True, "已存在，跳过"
    last_err = ""
    for attempt in range(retries):
        try:
            req = Request(url, headers=UA)
            with urlopen(req, timeout=90) as resp, dest.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return True, ""
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if dest.exists():
                dest.unlink(missing_ok=True)
            time.sleep(1 + attempt)
    return False, last_err


def process_dataset(ndjson: Path, out_root: Path, layout: str, workers: int):
    meta, images = read_records(ndjson)
    ds_name = sanitize(meta.get("name") or ndjson.stem)
    names = {int(k): v for k, v in meta.get("class_names", {}).items()}
    base = out_root / ds_name
    base.mkdir(parents=True, exist_ok=True)

    tasks = []
    split_count = {s: 0 for s in SPLITS}
    for i, rec in enumerate(images):
        if rec.get("type", "image") != "image" or not rec.get("file") or not rec.get("url"):
            continue
        split = rec.get("split", "train")
        if split not in SPLITS:
            split = "train"
        split_count[split] += 1

        suffix = Path(rec["file"]).suffix.lower() or ".jpg"
        stem = f"img_{i:06d}"
        if layout == "flat":
            img_dir = base / "flat" / split
            lbl_dir = img_dir
        else:
            img_dir = base / "images" / split
            lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        (lbl_dir / f"{stem}.txt").write_text(label_lines(rec), encoding="utf-8")
        tasks.append((rec["url"], img_dir / f"{stem}{suffix}", stem))

    print(f"\n=== {ds_name} ===")
    print(f"图片 {len(tasks)} 张 | 划分: " + ", ".join(f"{k}={v}" for k, v in split_count.items() if v))
    print(f"类别 {len(names)} 类: {names}")
    print(f"输出目录: {base}")

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, url, dest): stem for url, dest, stem in tasks}
        for fut in as_completed(futures):
            stem = futures[fut]
            ok, err = fut.result()
            done += 1
            if not ok:
                failed.append(f"{stem}\t{err}")
            if done % 500 == 0 or done == len(tasks):
                print(f"  已处理 {done}/{len(tasks)}，失败 {len(failed)}")

    if failed:
        (base / "download_failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"警告: {len(failed)} 张下载失败，明细见 {base / 'download_failed.txt'}")
    else:
        print("全部图片下载成功。")

    img_root = "flat" if layout == "flat" else "images"
    yaml = {
        "path": str(base.resolve()).replace("\\", "/"),
        "train": f"{img_root}/train",
        "val": f"{img_root}/val",
        "names": {str(k): v for k, v in names.items()},
    }
    if split_count["test"]:
        yaml["test"] = f"{img_root}/test"
    yaml_path = base / "data.yaml"
    head = [f"{k}: {v}" for k, v in yaml.items() if k != "names"]
    tail = ["names:"] + [f"  {k}: {v}" for k, v in yaml["names"].items()]
    yaml_path.write_text("\n".join(head + tail) + "\n", encoding="utf-8")
    print(f"已生成 {yaml_path}")
    return base


def main():
    p = argparse.ArgumentParser(description="把 Ultralytics NDJSON 下载成本地图片+YOLO标签文件夹")
    p.add_argument("--ndjson", action="append", required=True, help="NDJSON 文件路径，可重复传入多个")
    p.add_argument("--out", default=str(Path.home() / "Desktop" / "elevator_ebike_dataset"),
                   help="桌面上的输出文件夹，默认 ~/Desktop/elevator_ebike_dataset")
    p.add_argument("--layout", choices=["flat", "yolo"], default="flat",
                   help="flat=图片和txt同名放在同一文件夹（类似你的截图）；yolo=images/ + labels/ 标准结构")
    p.add_argument("--workers", type=int, default=16, help="并发下载线程数，默认16")
    args = p.parse_args()

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"输出根目录: {out_root}")
    print(f"目录布局: {args.layout} | 并发: {args.workers}")

    for ndjson in args.ndjson:
        process_dataset(Path(ndjson).expanduser(), out_root, args.layout, args.workers)

    print("\n全部完成。可在资源管理器中打开：", out_root)


if __name__ == "__main__":
    main()
