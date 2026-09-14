import os
from pathlib import Path

# 必须在导入 vLLM 前设置。当前环境的 FlashInfer 采样 JIT 存在
# CUDA 编译器与头文件版本冲突，因此改用 vLLM 原生采样器。
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main() -> None:
    model_path = Path("~/hf-mirror/Qwen/Qwen3.5-4B").expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"模型目录无效，未找到：{model_path / 'config.json'}"
        )

    model = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        use_fast=False,
        local_files_only=True,
    )

    messages = [{"role": "user", "content": "你是谁？"}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # 官方推荐的非思考模式采样参数。
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=128,
        presence_penalty=1.5,
    )
    llm = LLM(
        model=model,
        max_model_len=4096,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
    )
    outputs = llm.generate([text], sampling_params)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    # WSL 中 vLLM 使用 spawn 创建工作进程，模型初始化必须放在 main 保护中。
    main()
