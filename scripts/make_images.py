#!/usr/bin/env python3
"""生成项目展示图片：社交预览图(og-image.png) 与 README 头图(banner.png)。"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONT = "C:/Windows/Fonts/msyhbd.ttc"
FONT_REG = "C:/Windows/Fonts/msyh.ttc"


def font(size, bold=True):
    return ImageFont.truetype(FONT if bold else FONT_REG, size)


def linear_gradient(size, c1, c2, vertical=True):
    w, h = size
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(h):
        for x in range(w):
            t = (y / h) if vertical else (x / w)
            px[x, y] = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
    return img


def paste_cover(base, im, box, radius=14):
    """把图片裁剪填满 box 并圆角贴到 base 上。"""
    x, y, w, h = box
    ratio = max(w / im.width, h / im.height)
    nw, nh = int(im.width * ratio), int(im.height * ratio)
    im2 = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    im2 = im2.crop((left, top, left + w, top + h))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=255)
    base.paste(im2, (x, y), mask)


def paste_contain(base, im, box, bg=(255, 255, 255)):
    """把图片等比缩放完整放进 box，不裁切，居中显示，带圆角。"""
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    # 圆角背景
    panel = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rounded_rectangle([left, top, right, bottom], 20, fill=bg + (255,))
    base.paste(panel, (0, 0), panel)
    # 等比缩放到框内（留边距）
    pad = 12
    ratio = min((w - 2 * pad) / im.width, (h - 2 * pad) / im.height)
    nw, nh = max(1, int(im.width * ratio)), max(1, int(im.height * ratio))
    im2 = im.resize((nw, nh), Image.LANCZOS)
    px, py = left + (w - nw) // 2, top + (h - nh) // 2
    mask = Image.new("L", (nw, nh), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, nw - 1, nh - 1], 10, fill=255)
    base.paste(im2, (px, py), mask)


def make_og():
    """社交预览图 1200x630（Open Graph / Twitter Card）—— 左文右图布局。"""
    W, H = 1200, 630
    img = linear_gradient((W, H), (13, 27, 62), (26, 58, 120))
    d = ImageDraw.Draw(img)

    # 背景装饰光晕
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([W - 380, -160, W + 120, 340], fill=(56, 189, 248, 55))
    gd.ellipse([-160, H - 320, 240, H + 120], fill=(99, 102, 241, 45))
    glow = glow.filter(ImageFilter.GaussianBlur(60))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    d = ImageDraw.Draw(img)

    # ============ 左半区：文字 + 指标 ============
    # 顶部小标签
    d.rounded_rectangle([48, 46, 288, 88], 20, fill=(30, 64, 130))
    d.text((68, 55), "YOLO11n  ·  目标检测", font=font(22), fill=(147, 197, 253))

    # 主标题（两行，完整不溢出）
    d.text((46, 118), "电梯内电动车", font=font(60), fill=(255, 255, 255))
    d.text((46, 196), "智能检测系统", font=font(60), fill=(255, 255, 255))
    d.text((48, 288), "Elevator E-Bicycle Detection", font=font(24, False), fill=(148, 163, 184))

    # 三个指标卡（横排紧凑）
    stats = [("13,992", "训练图像"), ("5", "检测类别"), ("2", "对比实验")]
    x = 48
    for val, lab in stats:
        d.rounded_rectangle([x, 355, x + 152, 452], 14, fill=(255, 255, 255))
        d.text((x + 16, 366), val, font=font(38), fill=(29, 78, 216))
        d.text((x + 17, 415), lab, font=font(18, False), fill=(100, 116, 139))
        x += 164

    # 底部强调语
    d.text((48, 500), "高精度  ·  实时检测  ·  电梯安全", font=font(24), fill=(96, 165, 250))
    d.text((48, 552), "基于 Ultralytics YOLO11n", font=font(19, False), fill=(148, 163, 184))

    # ============ 右半区：一整块高亮检测图（完整显示，不裁切）============
    hero = ASSETS / "hero_detect.png"
    if hero.exists():
        paste_contain(img, Image.open(hero).convert("RGB"),
                      (560, 46, 1160, 590), bg=(255, 255, 255))

    out = ASSETS / "og-image.png"
    img.save(out, quality=95)
    print("saved", out, img.size)


def make_banner():
    """README 顶部头图 1600x560。"""
    W, H = 1600, 560
    img = linear_gradient((W, H), (248, 250, 255), (224, 235, 255), vertical=False)
    d = ImageDraw.Draw(img)

    # 左侧文字
    d.rounded_rectangle([64, 60, 520, 108], 24, fill=(29, 78, 216))
    d.text((88, 69), "SMART SAFETY · COMPUTER VISION", font=font(24), fill=(255, 255, 255))

    d.text((60, 140), "电梯内电动车智能检测系统", font=font(66), fill=(15, 23, 42))
    d.text((62, 236), "基于 YOLO11n 的电动车 / 自行车 / 婴儿车 / 行人 / 轮椅检测与识别", font=font(26, False), fill=(71, 85, 105))

    chips = ["YOLO11n", "Ultralytics Platform", "PyTorch", "ONNX Ready", "CC BY 4.0"]
    x = 62
    for c in chips:
        w = d.textlength(c, font=font(20, False)) + 32
        d.rounded_rectangle([x, 300, x + w, 348], 18, outline=(148, 163, 184), width=2, fill=(255, 255, 255))
        d.text((x + 16, 310), c, font=font(20, False), fill=(51, 65, 85))
        x += w + 14

    d.text((62, 400), "13,992 张图像   ·   5 类目标   ·   双数据集对比实验", font=font(24), fill=(29, 78, 216))

    # 右侧效果图
    demo = ASSETS / "demo_ds1_02.jpg"
    if demo.exists():
        paste_contain(img, Image.open(demo).convert("RGB"), (900, 55, 1545, 505), bg=(232, 238, 250))

    out = ASSETS / "banner.png"
    img.save(out, quality=95)
    print("saved", out, img.size)


if __name__ == "__main__":
    make_og()
    make_banner()
