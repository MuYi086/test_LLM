#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/muyi086/hf-mirror/Qwen/Qwen3.5-4B}"

cd "$PROJECT_DIR"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "模型目录无效，未找到：$MODEL_DIR/config.json" >&2
  echo "可通过 MODEL_DIR=/模型路径 ./start-vllm.sh 指定其他目录。" >&2
  exit 1
fi

# 当前环境的 FlashInfer 采样 JIT 存在 CUDA 编译器/头文件版本冲突。
# 本地尝鲜改用 vLLM 原生采样器，不影响 OpenAI 兼容 API。
export VLLM_USE_FLASHINFER_SAMPLER=0

exec uv run vllm serve "$MODEL_DIR" \
  --served-model-name Qwen3.5-4B \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8391
