#!/usr/bin/env python3
"""
把你从 Ultralytics Platform 下载的两个 .ndjson 数据集，
变成可以在电脑上直接展示给老师看的本地文件夹：

  demo/
    index.html                   ← 双击打开这个网页，原图 / 标注图并排展示
    ds1_elevator/
      originals/train/*.jpg      ← 原始照片
      annotated/train/*.jpg      ← 画了检测框和类别名的标注后照片
      labels/train/*.txt         ← YOLO 格式标签（备份）
    ds2_liftebike/
      ...

依赖：pip install pillow

用法：
  python scripts/build_demo_from_ndjson.py
  python scripts/build_demo_from_ndjson.py --max-images 120   # 每个数据集只取部分图片做展示
"""

import argparse
import html
import json
import shutil
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = [
    (255, 56, 56), (46, 204, 113), (52, 152, 219), (241, 196, 15),
    (155, 89, 182), (230, 126, 34), (26, 188, 156), (127, 140, 141),
]


def load_records(path):
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    records = [json.loads(ln) for ln in lines]
    meta = records[0]
    names = {int(k): v for k, v in meta.get("class_names", {}).items()}
    return meta, names, records[1:]


def color_for(cid):
    return COLORS[cid % len(COLORS)]


def draw_annotated(src_path, dst_path, boxes, names):
    """boxes: NDJSON 的 [class_id, xc, yc, w, h]（归一化 xywh）"""
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, w // 60))
    except OSError:
        font = ImageFont.load_default()

    for box in boxes:
        cid = int(box[0])
        xc, yc, bw, bh = map(float, box[1:5])
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))
        color = color_for(cid)
        label = names.get(cid, f"class{cid}")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, w // 300))
        # 标签底色
        try:
            tw, th = draw.textsize(label, font=font)
        except AttributeError:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 6], fill=color)
        draw.text((x1 + 5, ty + 2), label, fill=(255, 255, 255), font=font)

    img.save(dst_path, quality=92)


def process_dataset(recs, names, tag, out_dir, max_images):
    originals = out_dir / tag / "originals"
    annotated = out_dir / tag / "annotated"
    labels = out_dir / tag / "labels"
    for split in ("train", "val", "test"):
        for root in (originals, annotated, labels):
            (root / split).mkdir(parents=True, exist_ok=True)

    selected = []
    per_split = Counter()
    for r in recs:
        if r.get("type", "image") != "image" or not r.get("url"):
            continue
        split = r.get("split", "train")
        if split not in ("train", "val", "test"):
            split = "train"
        if max_images and per_split[split] >= max_images:
            continue
        per_split[split] += 1
        selected.append((r, split))

    print(f"[{tag}] 计划下载 {len(selected)} 张图（train/val/test={dict(per_split)}）")

    def one(item):
        r, split = item
        fname = Path(str(r["file"])).name
        src = originals / split / fname
        dst = annotated / split / fname
        lbl = labels / split / (Path(fname).stem + ".txt")
        boxes = (r.get("annotations") or {}).get("boxes", [])
        try:
            urllib.request.urlretrieve(r["url"], src)
        except Exception as e:
            return None, f"下载失败 {fname}: {e}"
        try:
            draw_annotated(src, dst, boxes, names)
        except Exception as e:
            return None, f"画框失败 {fname}: {e}"
        with lbl.open("w", encoding="utf-8") as f:
            for b in boxes:
                f.write(" ".join(str(x) for x in b) + "\n")
        return (tag, split, fname, fname, len(boxes), str(src.relative_to(out_dir)),
                str(dst.relative_to(out_dir))), None

    items, errors = [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(one, it) for it in selected]
        for fu in as_completed(futures):
            result, err = fu.result()
            if result:
                items.append(result)
            else:
                errors.append(err)
    if errors:
        print(f"[{tag}] 有 {len(errors)} 张图失败（可能是签名 URL 过期），示例: {errors[0]}")
    items.sort(key=lambda x: (x[1], x[2]))
    return items, names


def make_html(sections, out_dir):
    cards = []
    for tag, title, desc, items, names in sections:
        legend = "".join(
            f'<span class="tag" style="border-color:rgb{color_for(cid)};color:rgb{color_for(cid)}">'
            f'{cid} {html.escape(nm)}</span>'
            for cid, nm in sorted(names.items())
        )
        blocks = []
        for _, split, fname, _, nbox, src, dst in items:
            blocks.append(f"""
            <div class="card">
              <div class="pair">
                <figure><img src="{html.escape(src)}" loading="lazy"><figcaption>原图 · {html.escape(split)}</figcaption></figure>
                <figure><img src="{html.escape(dst)}" loading="lazy"><figcaption>标注后 · {nbox} 个目标</figcaption></figure>
              </div>
              <div class="fn">{html.escape(fname)}</div>
            </div>""")
        cards.append(f"""
        <section>
          <h2>{html.escape(title)}</h2>
          <p class="desc">{html.escape(desc)} · 共展示 {len(items)} 张（原图 / 标注后并排）</p>
          <div class="legend">{legend}</div>
          <div class="grid">{''.join(blocks) if blocks else '<p>没有成功下载的图片，请检查签名 URL 是否过期。</p>'}</div>
        </section>""")

    doc = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>电梯内电动车数据集展示</title>
<style>
 body{font-family:"Microsoft YaHei",sans-serif;margin:0;background:#f5f7fa;color:#1f2937}
 header{background:#1d4ed8;color:#fff;padding:24px 32px}
 header h1{margin:0 0 6px;font-size:26px} header p{margin:0;opacity:.9}
 section{padding:20px 32px} h2{font-size:21px;border-left:5px solid #1d4ed8;padding-left:10px}
 .desc{color:#6b7280;font-size:14px} .legend{margin:10px 0 18px}
 .tag{display:inline-block;border:2px solid;border-radius:16px;padding:2px 10px;margin:0 6px 6px 0;font-size:13px;background:#fff}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(620px,1fr));gap:18px}
 .card{background:#fff;border-radius:10px;padding:12px;box-shadow:0 2px 10px rgba(0,0,0,.06)}
 .pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
 figure{margin:0} img{width:100%;height:260px;object-fit:contain;background:#111;border-radius:6px}
 figcaption{font-size:12px;color:#6b7280;text-align:center;margin-top:4px}
 .fn{font-size:11px;color:#9ca3af;margin-top:6px;word-break:break-all}
</style></head><body>
<header><h1>基于 YOLO11n 的电梯内电动车检测 — 数据集展示</h1>
<p>左侧为原始照片，右侧为带检测框与类别名的标注后照片 · 本地离线打开即可</p></header>
""" + "".join(cards) + "</body></html>"

    path = out_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    return path


def main():
    p = argparse.ArgumentParser(description="从 NDJSON 生成可直接展示的照片/标注图文件夹")
    p.add_argument("--ds1", default="data/electric-bicycle-entering-elevator.ndjson")
    p.add_argument("--ds2", default="data/lift-ebike.ndjson")
    p.add_argument("--out", default="demo")
    p.add_argument("--max-images", type=int, default=100, help="每个数据集每个划分最多下载多少张，0=全部")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    specs = [
        (args.ds1, "ds1_elevator", "数据集 1：electric-bicycle-entering-elevator",
         "8193 张，5 类：婴儿车 / 电动车 / 自行车 / 人 / 轮椅"),
        (args.ds2, "ds2_liftebike", "数据集 2：lift-Ebike（电梯电动车）",
         "5799 张，1 类：motorcycle（电动车/摩托车）"),
    ]
    for path, tag, title, desc in specs:
        if not Path(path).is_file():
            print(f"跳过 {tag}: 找不到 {path}")
            continue
        meta, names, recs = load_records(path)
        items, _ = process_dataset(recs, names, tag, out, args.max_images)
        sections.append((tag, title, desc, items, names))

    if not sections:
        raise SystemExit("没有找到任何 NDJSON 文件，请先把两个 .ndjson 放到 data/ 目录")

    index = make_html(sections, out)
    print(f"\n展示页面已生成: {index.resolve()}")
    print("双击 index.html 即可给老师展示（原图与标注后照片并排）")


if __name__ == "__main__":
    main()
