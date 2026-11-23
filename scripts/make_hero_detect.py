#!/usr/bin/env python3
"""
生成一张「高亮检测图」：从原始照片 + YOLO 标签重新渲染，
检测框加粗、标签放大加实心底色，专门用于社交预览图/README 头图。

输出：assets/hero_detect.png（透明底的框会画在图上，整图输出）
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONT_BD = "C:/Windows/Fonts/msyhbd.ttc"
FONT_RG = "C:/Windows/Fonts/msyh.ttc"

# 类别中文名（用于标签显示，比英文更醒目）
LABEL_CN = {
    "electric-bicycle-entering-elevator": {
        0: "婴儿车", 1: "电动车", 2: "自行车", 3: "行人", 4: "轮椅",
    },
    "lift-Ebike": {0: "电动车"},
}
COLORS = [(46, 204, 113), (255, 56, 56), (52, 152, 219),
          (241, 196, 15), (155, 89, 182)]


def f(size, bold=True):
    return ImageFont.truetype(FONT_BD if bold else FONT_RG, size)


def parse_yolo(p):
    boxes = []
    if not p.is_file():
        return boxes
    for ln in p.read_text(encoding="utf-8").splitlines():
        q = ln.split()
        if len(q) < 5:
            continue
        try:
            boxes.append((int(float(q[0])), *map(float, q[1:5])))
        except ValueError:
            pass
    return boxes


def draw_highlight(src, boxes, cn_names, crop=None, out_name="hero_detect.png"):
    """在原图上画加粗框+大标签，可选裁剪聚焦目标区域。"""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)

    line_w = max(6, w // 180)          # 加粗框线
    font = f(max(30, w // 42))         # 大标签字体

    for cid, xc, yc, bw, bh in boxes:
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))
        color = COLORS[cid % len(COLORS)]
        label = cn_names.get(cid, f"class{cid}")

        # 半透明填充 + 加粗描边
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([x1, y1, x2, y2], fill=color + (38,))
        img = Image.alpha_composite(img.convert("RGBA"), overlay)
        d = ImageDraw.Draw(img)
        d.rectangle([x1, y1, x2, y2], outline=color + (255,), width=line_w)

        # 标签底色块 + 白字（放大）
        bb = d.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad_x, pad_y = 16, 10
        ty = y1 - th - 2 * pad_y if y1 - th - 2 * pad_y > 0 else y1 + 4
        tx2 = min(w, x1 + tw + 2 * pad_x)
        d.rounded_rectangle([x1, ty, tx2, ty + th + 2 * pad_y], 8, fill=color)
        d.text((x1 + pad_x, ty + pad_y - bb[1]), label,
               font=font, fill=(255, 255, 255))

    if crop:
        img = img.crop(crop)
    out = ASSETS / out_name
    img.convert("RGB").save(out, quality=96)
    print("saved", out, img.size)


def main():
    root = Path("D:/检测")
    ds = "electric-bicycle-entering-elevator"
    base = root / ds / "flat" / "train"
    # 用 img_000005：电动车 + 行人两张框，画面聚焦
    src = base / "img_000005.jpg"
    boxes = parse_yolo(base / "img_000005.txt")
    cn = LABEL_CN[ds]
    # 裁剪掉左侧电梯内壁空白，聚焦目标（原图1920x1080）
    draw_highlight(src, boxes, cn, crop=(700, 20, 1900, 1080),
                   out_name="hero_detect.png")


if __name__ == "__main__":
    main()
