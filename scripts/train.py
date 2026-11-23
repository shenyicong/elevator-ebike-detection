#!/usr/bin/env python3
"""
在 AMD Radeon Cloud (ROCm) 上训练 YOLO11n：
  实验1: 只用 electric-bicycle-entering-elevator 数据集
  实验2: 只用 lift-ebike 数据集
  实验3: 用合并后的 merged_elevator_ebike.ndjson

用法示例：
  python scripts/train.py --epochs 100 --batch 16 --device 0
  python scripts/train.py --experiments 3 --epochs 150 --amp-off   # 只跑合并集，关闭 AMP
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

EXPERIMENTS = {
    "1": {
        "name": "yolo11n_ds1_elevator",
        "data": "data/electric-bicycle-entering-elevator.ndjson",
        "desc": "电梯场景多类别数据集（婴儿车/电动车/自行车/人/轮椅）",
    },
    "2": {
        "name": "yolo11n_ds2_liftebike",
        "data": "data/lift-ebike.ndjson",
        "desc": "电梯电动车数据集（单类别 motorcycle）",
    },
    "3": {
        "name": "yolo11n_merged",
        "data": "data/merged_elevator_ebike.ndjson",
        "desc": "合并数据集（统一为 5 类）",
    },
}


def get_metric(res, *keys):
    d = getattr(res, "results_dict", {}) or {}
    for k in keys:
        if k in d:
            return d[k]
    return None


def main():
    p = argparse.ArgumentParser(description="YOLO11n 电梯内电动车检测三组训练")
    p.add_argument("--experiments", default="1,2,3", help="要运行的实验编号，逗号分隔，如 1,3")
    p.add_argument("--model", default="yolo11n.pt", help="预训练权重，默认 yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="ROCm 下仍用 0 / cuda:0，多卡如 0,1")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=20, help="早停 patience，0 表示不早停")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp-off", action="store_true", help="关闭混合精度（出现 NaN loss 或 mAP=0 时使用）")
    args = p.parse_args()

    ids = [x.strip() for x in args.experiments.split(",") if x.strip()]
    invalid = [x for x in ids if x not in EXPERIMENTS]
    if invalid:
        raise SystemExit(f"未知实验编号: {invalid}，可选: {list(EXPERIMENTS)}")

    rows = []
    for exp_id in ids:
        exp = EXPERIMENTS[exp_id]
        data_path = Path(exp["data"])
        if not data_path.is_file():
            print(f"跳过实验 {exp_id}: 找不到 {data_path}")
            continue

        print("\n" + "=" * 70)
        print(f"实验 {exp_id}: {exp['desc']}")
        print(f"数据: {data_path} | 模型: {args.model} | epochs={args.epochs} | batch={args.batch}")
        print("=" * 70)

        model = YOLO(args.model)  # 每次新建，避免不同类别数互相污染
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            amp=not args.amp_off,
            name=exp["name"],
            exist_ok=True,
        )

        val = model.val(
            data=str(data_path),
            split="val",
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
        )
        row = {
            "experiment": exp_id,
            "name": exp["name"],
            "dataset": str(data_path),
            "precision": get_metric(val, "metrics/precision(B)", "metrics/precision"),
            "recall": get_metric(val, "metrics/recall(B)", "metrics/recall"),
            "mAP50": get_metric(val, "metrics/mAP50(B)", "metrics/mAP50"),
            "mAP50-95": get_metric(val, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
            "weights": f"runs/detect/{exp['name']}/weights/best.pt",
        }
        rows.append(row)
        print("验证结果:", row)

    if rows:
        out = Path("training_summary.csv")
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n三组实验指标已保存: {out.resolve()}")


if __name__ == "__main__":
    main()
