#!/usr/bin/env python3
"""
推理后端基准测试：ONNX Runtime CPU vs OpenVINO CPU / Intel 核显。

回答一个问题：同一个 YOLO11n ONNX 模型，在不同后端上单张 640x640 推理要多久？

用法：
  pip install onnxruntime openvino pillow numpy
  python scripts/benchmark_inference.py

可选：
  python scripts/benchmark_inference.py --iter 200 --warmup 30
  python scripts/benchmark_inference.py --threads 4        # 只测 OpenVINO CPU 指定线程

输出：中位数 / 平均 / p95 延迟与 fps，并保存 benchmark_results.json。
对照浏览器 WASM 基线请用 scripts/benchmark_wasm.mjs（约 460ms/张）。
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "ds1_elevator": ROOT / "docs/assets/models/ds1_elevator.onnx",
    "ds2_liftebike": ROOT / "docs/assets/models/ds2_liftebike.onnx",
}
IMG = ROOT / "docs/assets/demo_ds1_02.jpg"
IMGSZ = 640


def make_input():
    im = Image.open(IMG).convert("RGB")
    iw, ih = im.size
    r = min(IMGSZ / iw, IMGSZ / ih)
    nw, nh = round(iw * r), round(ih * r)
    pad_x, pad_y = (IMGSZ - nw) / 2, (IMGSZ - nh) / 2
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (128, 128, 128))
    canvas.paste(im.resize((nw, nh)), (round(pad_x), round(pad_y)))
    return np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {
        "mean": round(statistics.mean(ts), 2),
        "median": round(statistics.median(ts), 2),
        "p95": round(ts[int(0.95 * len(ts)) - 1], 2),
        "fps": round(1000 / statistics.mean(ts), 1),
    }


def show(tag, r):
    print(f"  {tag:<26} mean={r['mean']:7.2f}ms  median={r['median']:7.2f}ms  "
          f"p95={r['p95']:7.2f}ms  ~{r['fps']:5.1f} fps")


def main():
    ap = argparse.ArgumentParser(description="YOLO11n 推理后端基准测试")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iter", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0, help="仅测 OpenVINO CPU 的指定线程数，0=默认全核")
    args = ap.parse_args()

    X = make_input()
    result = {}
    print(f"输入: {X.shape} ({X.nbytes / 1024 / 1024:.2f} MB), warmup={args.warmup}, iter={args.iter}")

    try:
        import onnxruntime as ort
        print(f"\n[ONNX Runtime {ort.__version__}] providers={ort.get_available_providers()}")
        for name, path in MODELS.items():
            opt = ort.SessionOptions()
            opt.intra_op_num_threads = 8
            opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0].name
            r = bench(lambda: sess.run(None, {inp: X}), args.warmup, args.iter)
            result.setdefault(name, {})["ort_cpu"] = r
            print(f"\n{name}:"); show("ORT CPU (8 threads)", r)
    except ImportError:
        print("\n[跳过] 未安装 onnxruntime: pip install onnxruntime")

    try:
        import openvino as ov
        core = ov.Core()
        print(f"\n[OpenVINO {ov.__version__}] devices={core.available_devices}")
        for name, path in MODELS.items():
            model = core.read_model(str(path))
            result.setdefault(name, {})
            print(f"\n{name}:")
            for dev in ["CPU", "GPU"]:
                if dev not in core.available_devices:
                    continue
                cfg = {"PERFORMANCE_HINT": "THROUGHPUT"}
                if dev == "CPU" and args.threads:
                    cfg["INFERENCE_NUM_THREADS"] = str(args.threads)
                try:
                    cm = core.compile_model(model, dev, cfg)
                    inp, out = cm.input(0), cm.output(0)
                    req = cm.create_infer_request()
                    r = bench(lambda: (req.infer({inp: X}), req.get_tensor(out))[1],
                              args.warmup, args.iter)
                    result[name]["ov_" + dev.lower()] = r
                    show(f"OpenVINO {dev}", r)
                except Exception as e:
                    print(f"  OpenVINO {dev} 失败: {e}")
    except ImportError:
        print("\n[跳过] 未安装 openvino: pip install openvino")

    print("\n" + "=" * 84)
    print("汇总 (median ms，越低越快)")
    print("=" * 84)
    print(f"{'模型':<16}{'ORT CPU':>12}{'OV CPU':>12}{'OV iGPU':>12}"
          f"{'OV/GPU vs ORT':>16}")
    for name, d in result.items():
        o = d.get("ort_cpu", {}).get("median", float("nan"))
        c = d.get("ov_cpu", {}).get("median", float("nan"))
        g = d.get("ov_gpu", {}).get("median", float("nan"))
        sp = f"{o / g:.2f}x" if g == g and g > 0 else "-"
        print(f"{name:<16}{o:12.2f}{c:12.2f}{g:12.2f}{sp:>16}")

    out = ROOT / "benchmark_results.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
