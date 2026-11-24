#!/usr/bin/env python3
"""
让 YOLO11 ONNX 能在浏览器 WebGPU 后端运行的模型改写脚本。

【问题】YOLO11 的检测头 DFL 模块里有一个 Softmax，它的 axis=1（非末轴）。
        ONNX Runtime Web 的 WebGPU 内核只支持末轴 Softmax，直接加载会报：
        "softmax only supports last axis for now"，导致自动降级到 CPU。

【做法】在该 Softmax 前后各插入一个 Transpose：
        原: [1,16,4,8400] --Softmax(axis=1)--> [1,16,4,8400]
        改: [1,16,4,8400] --Transpose--> [1,4,8400,16] --Softmax(axis=-1)-->
            [1,4,8400,16] --Transpose--> [1,16,4,8400]
        数学完全等价，输出逐元素一致（实测 maxdiff=0）。

【用法】pip install onnx
        python scripts/fix_webgpu_softmax.py 输入.onnx 输出.onnx
        改写后的模型同时兼容 WebGPU 和 CPU/WASM 两个后端，可直接替换原文件。
"""
import sys
import onnx
from onnx import helper

def fix(src, dst):
    m = onnx.load(src)
    g = m.graph
    target = None
    for n in g.node:
        if n.op_type == "Softmax" and any(a.name == "axis" and a.i == 1 for a in n.attribute):
            # 只处理 dfl 的那个（非注意力层，注意力层是 axis=-1）
            if "dfl" in n.name:
                target = n
                break
    if target is None:
        print(f"  {src}: 未找到 axis=1 的 dfl Softmax，跳过")
        onnx.save(m, dst)
        return

    old_in = target.input[0]
    old_out = target.output[0]
    # 前置 transpose: [1,16,4,8400] -> [1,4,8400,16]，把第1轴移到末轴
    pre_name = target.name + "_pre_transpose"
    pre_out = target.name + "_pre_transpose_out"
    pre = helper.make_node("Transpose", [old_in], [pre_out], name=pre_name, perm=[0, 2, 3, 1])
    # 后置 transpose: [1,4,8400,16] -> [1,16,4,8400]
    post_name = target.name + "_post_transpose"
    post_out = target.name + "_post_transpose_out"
    post = helper.make_node("Transpose", [old_out], [post_out], name=post_name, perm=[0, 3, 1, 2])

    target.input[0] = pre_out
    # 改 axis 为 -1
    for a in target.attribute:
        if a.name == "axis":
            a.i = -1

    # 把原来消费 softmax 输出的节点改成消费 post_out
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp == old_out:
                n.input[i] = post_out

    # 插入到 softmax 之前（onnx 拓扑顺序：pre, softmax, post）
    nodes = list(g.node)
    idx = nodes.index(target)
    nodes[idx:idx + 1] = [pre, target, post]
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"  {src} -> {dst}: 已插入 pre/post Transpose, Softmax axis -> -1")

if __name__ == "__main__":
    fix(sys.argv[1], sys.argv[2])
