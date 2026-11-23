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


def make_og():
    """社交预览图 1200x630（Open Graph / Twitter Card）。"""
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

    # 顶部小标签
    d.rounded_rectangle([56, 48, 300, 88], 20, fill=(30, 64, 130))
    d.text((76, 57), "YOLO11n  ·  目标检测", font=font(22), fill=(147, 197, 253))

    # 主标题
    d.text((54, 118), "电梯内电动车智能检测系统", font=font(62), fill=(255, 255, 255))
    d.text((56, 205), "Elevator E-Bicycle Detection · YOLO11n", font=font(28, False), fill=(148, 163, 184))

    # 三个指标卡
    stats = [("13,992", "训练图像"), ("5", "检测类别"), ("2", "对比实验")]
    x = 56
    for val, lab in stats:
        d.rounded_rectangle([x, 285, x + 250, 380], 16, fill=(255, 255, 255))
        d.text((x + 22, 297), val, font=font(46), fill=(29, 78, 216))
        d.text((x + 24, 352), lab, font=font(20, False), fill=(100, 116, 139))
        x += 272

    # 底部：检测效果图拼贴
    demo = ASSETS / "demo_ds1_01.jpg"
    if demo.exists():
        paste_cover(img, Image.open(demo).convert("RGB"), (56, 410, 560, 600), 16)

    demo2 = ASSETS / "demo_ds2_02.jpg"
    if demo2.exists():
        paste_cover(img, Image.open(demo2).convert("RGB"), (580, 410, 1084, 600), 16)

    d = ImageDraw.Draw(img)
    d.text((56, 604 - 0), "", font=font(10))  # noqa
    out = ASSETS / "og-image.png"
    img.save(out, quality=95)
    print("saved", out, img.size)


def make_banner():
    """README 顶部头图 1600x560。"""
    W, H = 1600, 560
    img = linear_gradient((W, H), (248, 250, 255), (224, 235, 255), vertical=False)
    d = ImageDraw.Draw(img)

    # 左侧文字
    d.rounded_rectangle([64, 60, 500, 108], 24, fill=(29, 78, 216))
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
        paste_cover(img, Image.open(demo).convert("RGB"), (900, 55, 1545, 505), 20)

    out = ASSETS / "banner.png"
    img.save(out, quality=95)
    print("saved", out, img.size)


if __name__ == "__main__":
    make_og()
    make_banner()
