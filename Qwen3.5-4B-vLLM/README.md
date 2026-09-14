# Qwen3.5-4B vLLM（WSL）部署

本项目用于在 WSL 中通过 vLLM 启动本地的 `Qwen3.5-4B`，并提供 OpenAI 兼容 API。

依赖版本按 [Datawhale Qwen3.5-4B vLLM 部署文档](https://github.com/datawhalechina/self-llm/blob/master/models/Qwen3.5/01-Qwen3.5-4B-vLLM%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md) 记录在 `pyproject.toml`：Python 3.12、Torch 2.11.0、vLLM 0.23.0、Transformers 4.57+。

## 0. 在 WSL 中操作

请在 Ubuntu/WSL 终端中执行命令，而不是在 Git Bash 的 `//wsl.localhost/...` 路径下运行 Windows 版工具：

```bash
cd /home/muyi086/demo/testQwen3.5-4B
```

项目虚拟环境由 `uv` 管理。运行项目命令时始终使用 `uv run ...`；它会自动使用当前目录的 `.venv`，不需要额外执行 `source .venv/bin/activate`。

## Quick Start

依赖和模型已准备好时，在一个全新的 WSL 终端中执行：

```bash
cd /home/muyi086/demo/testQwen3.5-4B
./start-vllm.sh
```

脚本已经包含当前 16GB 显卡需要的兼容参数。服务出现 `Application startup complete` 后，另开一个 WSL 终端执行：

```bash
curl http://127.0.0.1:8391/v1/models
```

## 1. 安装完成后的版本检查

依赖下载可按自己的镜像或加速方式完成。安装结束后，先确认关键版本：

```bash
uv run python - <<'PY'
from importlib.metadata import version
import torch

for package in ("vllm", "flashinfer-python", "transformers", "torch"):
    print(f"{package}={version(package)}")
print(f"torch.cuda={torch.version.cuda}")
print(f"cuda.available={torch.cuda.is_available()}")
PY
```

本项目当前目标版本：

```text
vllm=0.23.0
torch=2.11.0
torchvision=0.26.0
torchaudio=2.11.0
transformers>=4.57
```

`cuda.available=True` 才表示 WSL 中的 PyTorch 能访问显卡。

## 2. 确认模型目录

默认使用以下本地模型目录：

```bash
MODEL_DIR=/home/muyi086/hf-mirror/Qwen/Qwen3.5-4B
test -f "$MODEL_DIR/config.json" && echo "模型目录正常"
```

目录中至少应有 `config.json`、`tokenizer.json` 和模型的 `.safetensors` 权重文件。

## 3. 当前 WSL 环境的 FlashInfer 兼容性处理

当前依赖版本与 Datawhale 文档一致：Torch 2.11.0（CUDA 13.0）和 vLLM 0.23.0。模型也已能够成功加载。但 `flashinfer-python==0.6.12` 会在服务预热时即时编译 top-k/top-p 采样算子，而本机环境中存在两套不一致的 CUDA 组件：

- 虚拟环境中的 `nvcc` 是 CUDA 13.2；
- Torch/vLLM 依赖的 `nvidia/cu13/include` 头文件是 CUDA 13.0。

因此，**不要**再通过设置 `CUDA_HOME`、`CUDACXX` 或修改 `PATH` 来强制 FlashInfer 使用虚拟环境的 CUDA 13.2 编译器。这会导致下面的确定性错误：

```text
CUDA compiler and CUDA toolkit headers are incompatible
```

若此前设置过上述环境变量，建议关闭当前 WSL 终端并新开一个终端，再回到项目目录：

```bash
cd /home/muyi086/demo/testQwen3.5-4B
```

本项目采用 vLLM 内置的原生采样器，跳过有兼容性问题的 FlashInfer 采样 JIT：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_FLASHINFER_SAMPLER
```

这不会影响模型加载、OpenAI API 格式或生成结果的正确性；仅可能使 top-k/top-p 采样的性能略低于 FlashInfer 加速版本。模型的 FlashAttention 路径不受此设置影响。

## 4. 启动服务

RTX 4070 Ti SUPER 为 16GB 显存。相比参考文档的 24GB 显卡，本项目使用 `--gpu-memory-utilization 0.8`，并将本地测试并发限制为 8。

推荐直接运行一键脚本：

```bash
./start-vllm.sh
```

脚本实际执行的完整命令如下：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 uv run vllm serve /home/muyi086/hf-mirror/Qwen/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8391
```

其中：

- `--max-num-seqs 8`：默认值 256 超过当前可用的 143 个 Mamba cache block；本地测试限制为 8 即可。
- `--enforce-eager`：关闭 CUDA Graph 捕获，避免 Mamba cache 与 CUDA Graph 的启动限制，并减少首次启动等待。

出现以下日志即表示服务启动成功：

```text
Application startup complete.
```

## 5. 验证 API

保持服务终端不要退出，在另一个 WSL 终端执行：

```bash
curl http://127.0.0.1:8391/v1/models
```

再发送文本对话请求：

```bash
curl http://127.0.0.1:8391/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-4B",
    "messages": [{"role": "user", "content": "用一句话介绍深度学习。"}],
    "max_tokens": 128,
    "temperature": 0.2,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

注意：上面是直接使用 `curl` 发送 HTTP JSON，因此 `chat_template_kwargs` 位于请求体顶层。只有使用 Python `openai` 客户端时，才需要通过客户端的 `extra_body` 参数传递它。

## 6. Python 离线推理

不启动 API 服务、直接在 Python 进程中加载本地模型：

```bash
cd /home/muyi086/demo/testQwen3.5-4B
uv run vllm_model.py
```

`vllm_model.py` 已处理以下事项：

- 使用 `Path.expanduser()` 将 `~/hf-mirror/...` 转换成绝对路径，避免被 Transformers 误判为 Hugging Face 仓库 ID；
- 设置 `local_files_only=True`，只读取本地 tokenizer 文件；
- 禁用当前环境中无法编译的 FlashInfer 采样 JIT；
- 将并发限制为 8，并启用 eager 模式，适配当前 16GB 显卡。
- 使用 `if __name__ == "__main__"` 入口，兼容 WSL 下 vLLM 的 `spawn` 多进程模式。

如需修改提示词，编辑脚本中的 `messages`；如需修改模型位置，编辑 `model_path`。

## 常见问题

### `vllm: command not found`

不要直接运行 `vllm serve ...`。请在项目目录运行：

```bash
uv run vllm serve ...
```

### `Repo id must be ... '~/hf-mirror/...'`

Python 不会像 shell 一样自动展开普通字符串中的 `~`。应使用 `Path(...).expanduser()` 后再传给 Transformers；本项目的 `vllm_model.py` 已修复此问题。

### `Free memory ... is less than desired GPU memory utilization`

显存空闲量低于 vLLM 要预留的比例。先用 `nvidia-smi` 检查其他任务；确保显存空闲后，将 `--gpu-memory-utilization` 从 `0.8` 继续调低到 `0.75`。

### `max_num_seqs (256) exceeds available Mamba cache blocks`

默认并发 256 高于 16GB 显卡能够分配的 Mamba cache block 数量。使用本项目的 `./start-vllm.sh` 即可；脚本已设置 `--max-num-seqs 8` 和 `--enforce-eager`。

### `BlockAdjacentDifference ... FlagHeads` 或 `CUDA compiler and CUDA toolkit headers are incompatible`

两种错误都发生在 FlashInfer 采样算子的即时编译阶段：前者是系统 CUDA 12 与 FlashInfer 的 CCCL/CUB 不兼容，后者是 CUDA 13.2 编译器与 CUDA 13.0 头文件不兼容。它们不是模型、显存或 `uv` 虚拟环境未激活造成的。

关闭当前终端后重新打开 WSL 终端，进入项目目录，并使用原生采样器启动：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 uv run vllm serve /home/muyi086/hf-mirror/Qwen/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8391
```

启动日志出现以下提示即表示 FlashInfer 采样器已被正确跳过：

```text
FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.
```

### Transformers v4 或 WSL `pin_memory=False` 警告

这两项都是警告，不是本项目当前的启动失败原因。FlashInfer 编译错误应按第 3 节处理；显存错误应按上一节处理。
