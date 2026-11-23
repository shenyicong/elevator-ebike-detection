#!/usr/bin/env bash
# 在 AMD Radeon Cloud 实例上运行：检查/安装 ROCm 版 PyTorch 与 Ultralytics
set -e

echo "===== 1. 检查 GPU / ROCm 驱动 ====="
ls -l /dev/kfd /dev/dri 2>/dev/null || echo "警告: 未找到 /dev/kfd 或 /dev/dri，请确认实例已分配 AMD GPU"
grep -rhs '^gfx_target_version' /sys/class/kfd/kfd/topology/nodes 2>/dev/null | head

echo "===== 2. 检查 PyTorch 是否已有 ROCm 支持 ====="
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.version.hip else 1)" 2>/dev/null; then
  echo "已检测到可用的 ROCm PyTorch，跳过安装。"
else
  echo "未检测到 ROCm PyTorch，开始安装..."
  GPU_ARCH=$(grep -rhs '^gfx_target_version' /sys/class/kfd/kfd/topology/nodes 2>/dev/null \
    | awk '$2 {printf "device-gfx%d%d%x\n", int($2/10000), int($2/100)%100, $2%100}' \
    | sort -u | paste -sd, -)
  echo "检测到的 GPU 架构: ${GPU_ARCH:-未检测到}"
  if [ -z "$GPU_ARCH" ]; then
    echo "错误: 没有检测到 AMD GPU 架构，请检查实例规格/镜像。" >&2
    exit 1
  fi
  pip install "torch[${GPU_ARCH}]" "torchvision[${GPU_ARCH}]" \
    --index-url https://stable.repo.amd.com/rocm/whl-next/
fi

echo "===== 3. 安装 / 升级 Ultralytics ====="
pip install -U ultralytics

echo "===== 4. 环境验证 ====="
python - <<'PY'
import torch, ultralytics
print("Python torch:", torch.__version__)
print("HIP version :", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name     :", torch.cuda.get_device_name(0))
print("Ultralytics  :", ultralytics.__version__)
PY

echo "===== 完成 ====="
echo "若上面 GPU available 为 True，即可继续运行 scripts/train.py"
