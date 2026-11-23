#!/usr/bin/env python3
"""
无需登录即可生成一个小规模展示样例：
从两个数据集公开页面抓取示例图片及标注，下载到本地并画出检测框，
生成 public_demo/index.html，双击即可给老师看“原图 / 标注后照片”对比。

注意：公开页面只提供少量样例图（每集 4 张高清 + 若干缩略图）。
若要展示完整数据集，请先登录平台下载 .ndjson，再用 build_demo_from_ndjson.py。
"""

import html
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_demo_from_ndjson import draw_annotated, make_html  # noqa: E402

PAGES = [
    ("ds1_elevator", "数据集 1：electric-bicycle-entering-elevator",
     "8193 张 · 5 类：婴儿车 / 电动车 / 自行车 / 人 / 轮椅",
     "https://platform.ultralytics.com/kingfisher-panther/datasets/electric-bicycle-entering-elevator",
     {0: "Baby-carriage", 1: "Electric-bicycle", 2: "bicycle", 3: "person", 4: "wheelchair"}),
    ("ds2_liftebike", "数据集 2：lift-Ebike（电梯电动车）",
     "5799 张 · 1 类：motorcycle（电动车/摩托车）",
     "https://platform.ultralytics.com/w-xiao/datasets/lift-ebike",
     {0: "motorcycle"}),
]


def extract_array(s, key):
    """从 Next.js 转义 HTML 中抽取 \"key\":[...] 数组。"""
    # 先把整段转义 JSON 还原成普通 JSON 文本，再用标准解码器精确切出数组
    unescaped = s.replace('\\"', '"').replace('\\u0026', '&').replace('\\/', '/')
    token = '"' + key + '":['
    i = unescaped.find(token)
    if i < 0:
        return []
    j = i + len(token) - 1
    try:
        obj, _ = json.JSONDecoder().raw_decode(unescaped[j:])
        return obj
    except json.JSONDecodeError as e:
        print("JSON 解析失败:", e)
        return []


def main():
    out = Path("public_demo")
    out.mkdir(parents=True, exist_ok=True)
    sections = []

    for tag, title, desc, url, names in PAGES:
        print("获取页面:", title)
        page = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore")
        samples = extract_array(page, "sampleImages")
        initials = extract_array(page, "initialImages")
        print(f"  样例 {len(samples)} 张, 初始图 {len(initials)} 张")

        candidates = []
        for im in samples:
            if str(im.get("imageUrl", "")).startswith("http"):
                candidates.append((im, "高清样例"))
        for im in initials[:16]:
            if str(im.get("thumbnailUrl", "")).startswith("http"):
                candidates.append((im, im.get("split", "train")))

        (out / tag / "originals").mkdir(parents=True, exist_ok=True)
        (out / tag / "annotated").mkdir(parents=True, exist_ok=True)
        items = []
        for idx, (im, split) in enumerate(candidates):
            img_url = im["imageUrl"] if str(im.get("imageUrl", "")).startswith("http") else im["thumbnailUrl"]
            ext = ".webp" if "webp" in img_url else ".jpg"
            fname = f"{tag}_{idx:02d}{ext}"
            src_rel = Path(tag) / "originals" / fname
            dst_rel = Path(tag) / "annotated" / fname
            src = out / src_rel
            dst = out / dst_rel
            # 公开页面标注是 {"classId": int, "bbox": [xc,yc,w,h]}，
            # 转成画框函数需要的 [cls,xc,yc,w,h]
            raw_labels = im.get("labels", [])
            boxes = []
            for lab in raw_labels:
                if isinstance(lab, dict) and "classId" in lab and "bbox" in lab:
                    boxes.append([lab["classId"], *lab["bbox"]])
                elif isinstance(lab, (list, tuple)):
                    boxes.append(list(lab))
            try:
                urllib.request.urlretrieve(img_url, src)
                draw_annotated(src, dst, boxes, names)
            except Exception as e:
                print(f"  跳过第 {idx} 张: {e}")
                continue
            items.append((tag, split, fname, fname, len(im.get("labels", [])),
                          src_rel.as_posix(), dst_rel.as_posix()))

        sections.append((tag, title, desc, items, names))

    index = make_html(sections, out)
    print(f"\n展示页面已生成: {index.resolve()}")
    print("双击打开即可（原图 / 标注后照片并排）")


if __name__ == "__main__":
    main()
