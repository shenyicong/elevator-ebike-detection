#!/usr/bin/env python3
"""
把本地已下载好的 YOLO 数据集（图片 + 同名 .txt 标签）直接渲染成
「标注后照片」，并生成一个原图 / 标注图并排的 HTML 画廊。

不依赖 X-AnyLabeling，不需要导入标签、不需要登录平台，
双击生成的 HTML 就能给老师展示。

输出结构（以 D:/检测 为例）：
    D:/检测/
    ├─ 展示_原图与标注图.html          ← 双击打开这个
    └─ electric-bicycle-entering-elevator/
       ├─ annotated/train/img_000000.jpg   ← 画好框的照片
       ├─ annotated/val/...
       └─ lift-Ebike/annotated/...

依赖：pip install pillow

用法：
    python scripts/render_annotated.py                       # 默认 root=D:/检测，每个划分取 60 张
    python scripts/render_annotated.py --root "D:/检测" --max 100
    python scripts/render_annotated.py --max 0               # 全部图片（慢，占空间）
"""

import argparse
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = [
    (255, 56, 56), (46, 204, 113), (52, 152, 219), (241, 196, 15),
    (155, 89, 182), (230, 126, 34), (26, 188, 156), (127, 140, 141),
]

# 要处理的数据集（目录名 -> 展示标题 / 说明）
DATASETS = [
    ("electric-bicycle-entering-elevator",
     "数据集 1：electric-bicycle-entering-elevator",
     "8193 张 · 5 类：婴儿车 / 电动车 / 自行车 / 人 / 轮椅"),
    ("lift-Ebike",
     "数据集 2：lift-Ebike（电梯电动车）",
     "5799 张 · 1 类：motorcycle（电动车/摩托车）"),
]


def read_classes(ds_dir):
    """优先读 classes.txt，否则从 data.yaml 的 names 里解析。"""
    p = ds_dir / "classes.txt"
    if p.exists():
        names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if names:
            return names
    p = ds_dir / "data.yaml"
    if p.exists():
        names = {}
        in_names = False
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip().startswith("names:"):
                in_names = True
                continue
            if in_names and ln.strip() and not ln.startswith(" "):
                break
            if in_names and ":" in ln:
                k, v = ln.strip().split(":", 1)
                try:
                    names[int(k.strip())] = v.strip().strip('"\'')
                except ValueError:
                    pass
        if names:
            return [names[i] for i in sorted(names)]
    raise RuntimeError(f"找不到类别定义：{ds_dir}")


def parse_yolo(txt_path):
    """读 YOLO 标签：每行 类别 x_center y_center width height（归一化）。"""
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


def draw_annotated(src_path, dst_path, boxes, names):
    """在图片上画检测框和类别名，保存为标注后照片。"""
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, w // 55))
    except OSError:
        font = ImageFont.load_default()
    line_w = max(2, w // 320)

    for cid, xc, yc, bw, bh in boxes:
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))
        color = COLORS[cid % len(COLORS)]
        label = names[cid] if 0 <= cid < len(names) else f"class{cid}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)
        ty = y1 - th - 8 if y1 - th - 8 > 0 else y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 6], fill=color)
        draw.text((x1 + 5, ty + 2), label, fill=(255, 255, 255), font=font)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path, quality=92)


def render_dataset(root, dirname, title, desc, max_per_split):
    ds_dir = root / dirname
    if not ds_dir.is_dir():
        print(f"跳过：找不到 {ds_dir}")
        return None
    names = read_classes(ds_dir)
    cards = []
    for split in ("train", "val", "test"):
        img_dir = ds_dir / "flat" / split
        if not img_dir.is_dir():
            continue
        out_dir = ds_dir / "annotated" / split
        images = sorted(img_dir.glob("*.jpg"))
        if max_per_split:
            images = images[:max_per_split]
        made = 0
        for img_path in images:
            lbl = img_dir / (img_path.stem + ".txt")
            boxes = parse_yolo(lbl)
            if not boxes:
                continue  # 没有目标的图不生成，避免展示空框
            dst = out_dir / img_path.name
            try:
                draw_annotated(img_path, dst, boxes, names)
            except Exception as e:
                print(f"  画框失败 {img_path.name}: {e}")
                continue
            made += 1
            rel_src = f"{dirname}/flat/{split}/{img_path.name}"
            rel_dst = f"{dirname}/annotated/{split}/{img_path.name}"
            cards.append((rel_src, rel_dst, split, img_path.name, len(boxes)))
        print(f"  {dirname}/{split}: 生成 {made} 张标注图 -> {out_dir}")

    legend = "".join(
        f'<span class="tag" style="border-color:rgb{COLORS[i % len(COLORS)]};'
        f'color:rgb{COLORS[i % len(COLORS)]}">{i} {html.escape(nm)}</span>'
        for i, nm in enumerate(names)
    )
    blocks = "".join(
        f"""<div class="card">
          <div class="pair">
            <figure><img src="{html.escape(src)}" loading="lazy"><figcaption>原图 · {html.escape(split)}</figcaption></figure>
            <figure><img src="{html.escape(dst)}" loading="lazy"><figcaption>标注后 · {nbox} 个目标</figcaption></figure>
          </div>
          <div class="fn">{html.escape(fname)}</div>
        </div>"""
        for src, dst, split, fname, nbox in cards
    )
    section = f"""
    <section>
      <h2>{html.escape(title)}</h2>
      <p class="desc">{html.escape(desc)} · 展示 {len(cards)} 张（左原图 / 右标注后）</p>
      <div class="legend">{legend}</div>
      <div class="grid">{blocks or '<p>没有可展示的图片。</p>'}</div>
    </section>"""
    return section


def make_html(sections, out_path):
    doc = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>电梯内电动车检测 — 原图与标注图展示</title>
<style>
 body{font-family:"Microsoft YaHei",sans-serif;margin:0;background:#f5f7fa;color:#1f2937}
 header{background:#1d4ed8;color:#fff;padding:22px 32px}
 header h1{margin:0 0 6px;font-size:25px} header p{margin:0;opacity:.92}
 section{padding:20px 32px} h2{font-size:20px;border-left:5px solid #1d4ed8;padding-left:10px}
 .desc{color:#6b7280;font-size:14px} .legend{margin:10px 0 16px}
 .tag{display:inline-block;border:2px solid;border-radius:16px;padding:2px 10px;margin:0 6px 6px 0;font-size:13px;background:#fff}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(600px,1fr));gap:18px}
 .card{background:#fff;border-radius:10px;padding:12px;box-shadow:0 2px 10px rgba(0,0,0,.06)}
 .pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
 figure{margin:0} img{width:100%;height:250px;object-fit:contain;background:#111;border-radius:6px}
 figcaption{font-size:12px;color:#6b7280;text-align:center;margin-top:4px}
 .fn{font-size:11px;color:#9ca3af;margin-top:6px;word-break:break-all}
</style></head><body>
<header><h1>基于 YOLO11n 的电梯内电动车检测 — 数据集展示</h1>
<p>左侧为原始照片，右侧为按 YOLO 标签画框后的照片 · 离线打开即可，无需标注软件</p></header>
""" + "".join(sections) + "</body></html>"
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def main():
    p = argparse.ArgumentParser(description="把本地 YOLO 数据集渲染成标注后照片 + 展示网页")
    p.add_argument("--root", default="D:/检测", help="数据集所在根目录")
    p.add_argument("--max", type=int, default=60, dest="max_per_split",
                   help="每个划分最多生成多少张标注图，0=全部")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"找不到目录：{root}")

    sections = []
    for dirname, title, desc in DATASETS:
        sec = render_dataset(root, dirname, title, desc, args.max_per_split)
        if sec:
            sections.append(sec)

    if not sections:
        raise SystemExit("没有找到任何数据集目录")

    index = root / "展示_原图与标注图.html"
    make_html(sections, index)
    print(f"\n展示页面已生成：{index}")
    print("双击这个 HTML 即可给老师展示（原图 / 标注后并排）")


if __name__ == "__main__":
    main()
