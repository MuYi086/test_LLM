"""用本地模型逐条运行固定提示词，并保存可用于 LoRA 对照的推理记录。"""

import argparse
import copy
import hashlib
import json
import os
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from check_environment import (
    PROJECT_ROOT,
    ModelConfig,
    check_gpu,
    check_local_model,
    check_python,
    load_model_config,
)

DEFAULT_PROMPTS = PROJECT_ROOT / "eval" / "prompts.jsonl"
DEFAULT_INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "inference.toml"


@dataclass(frozen=True)
class InferenceConfig:
    max_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    seed: int
    enable_thinking: bool
    device: str
    attn_impl: str


def load_inference_config(path: Path = DEFAULT_INFERENCE_CONFIG) -> InferenceConfig:
    with path.open("rb") as file:
        config = InferenceConfig(**tomllib.load(file)["inference"])
    if type(config.max_tokens) is not int or config.max_tokens <= 0:
        raise ValueError("max_tokens 必须是正整数")
    if config.temperature < 0 or not 0 < config.top_p <= 1 or config.repetition_penalty <= 0:
        raise ValueError("temperature / top_p / repetition_penalty 的取值无效")
    if type(config.seed) is not int or type(config.enable_thinking) is not bool:
        raise ValueError("seed 必须是整数，enable_thinking 必须是布尔值")
    if config.device != "cuda:0" or config.attn_impl != "sdpa":
        raise ValueError("第一轮固定使用 cuda:0 / sdpa，请保留这两个配置")
    return config


def load_prompts(path: Path = DEFAULT_PROMPTS) -> list[dict]:
    """读取独立的单轮测试输入，拒绝重复 ID 和混入的 assistant 答案。"""
    cases = []
    seen = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            identifier = case.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError(f"第 {line_number} 行的 id 缺失或重复")
            messages = case.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{identifier}: messages 必须是非空列表")
            roles = []
            for message in messages:
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    raise ValueError(f"{identifier}: 每条 message 需要字符串 content")
                roles.append(message.get("role"))
            if roles not in (["user"], ["system", "user"]):
                raise ValueError(f"{identifier}: 只接受 user 或 system+user，不允许混入目标答案")
            seen.add(identifier)
            cases.append(case)
    if not cases:
        raise ValueError(f"提示词文件为空：{path}")
    return cases


def prompts_fingerprint(cases: list[dict]) -> str:
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_engine(model: ModelConfig, config: InferenceConfig, adapter: Path | None = None):
    """此函数会真实加载权重到 GPU；只在你执行推理时调用。"""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_HF"] = "1"

    import torch
    from swift.infer_engine import TransformersEngine

    torch.manual_seed(config.seed)
    engine = TransformersEngine(
        str(model.path),
        model_type=model.model_type,
        torch_dtype=torch.bfloat16,
        device_map=config.device,
        attn_impl=config.attn_impl,
        max_batch_size=1,
        use_hf=True,
        model_kwargs={"local_files_only": True},
        adapters=[str(adapter)] if adapter else None,
    )
    engine.model.eval()
    return engine


def infer_case(engine, case: dict, config: InferenceConfig) -> dict:
    """一次只生成一条，保留输出、token 用量、结束原因和耗时。"""
    from swift.infer_engine import InferRequest, RequestConfig

    request = InferRequest(
        messages=copy.deepcopy(case["messages"]),
        chat_template_kwargs={"enable_thinking": config.enable_thinking},
    )
    request_config = RequestConfig(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        seed=config.seed,
        stream=False,
    )
    started = time.perf_counter()
    responses = engine.infer([request], request_config, use_tqdm=False)
    if len(responses) != 1 or not responses[0].choices:
        raise RuntimeError(f"{case['id']}: 推理引擎没有返回单条有效响应")
    response = responses[0]
    choice = response.choices[0]
    return {
        **copy.deepcopy(case),
        "output": choice.message.content,
        "reasoning_content": getattr(choice.message, "reasoning_content", None),
        "finish_reason": choice.finish_reason,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def write_run_results(
    engine, cases: list[dict], config: InferenceConfig, output_dir: Path, metadata: dict
) -> Path:
    """保留逐条已完成结果；拒绝覆盖已有运行目录。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = copy.deepcopy(metadata)
    metadata.update(status="running", completed_count=0, prompt_count=len(cases))
    metadata_path = output_dir / "metadata.json"

    def save_metadata():
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    save_metadata()
    results_path = output_dir / "results.jsonl"
    try:
        with results_path.open("x", encoding="utf-8") as file:
            for index, case in enumerate(cases, 1):
                print(f"\n[{index}/{len(cases)}] {case['id']}", flush=True)
                result = infer_case(engine, case, config)
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                metadata["completed_count"] = index
                save_metadata()
                print(result["output"], flush=True)
                print(
                    f"finish_reason={result['finish_reason']}, usage={result['usage']}", flush=True
                )
                if result["finish_reason"] == "length":
                    print("本条触及输出上限；请在人工检查中标记截断。", flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_metadata()
        raise
    metadata.update(status="complete", finished_at=datetime.now(UTC).isoformat())
    save_metadata()
    return results_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="models.toml 的配置名；默认 Qwen3.5-4B")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_INFERENCE_CONFIG)
    parser.add_argument("--limit", type=int, help="仅运行前 N 条，用于首次单条验证")
    parser.add_argument("--output-dir", type=Path, help="新运行目录；不覆盖已存在的目录")
    parser.add_argument("--adapter", type=Path, help="未来对照使用的本地 LoRA checkpoint")
    parser.add_argument(
        "--dry-run", action="store_true", help="只验证输入，不导入训练库、不加载模型"
    )
    args = parser.parse_args()
    model = load_model_config(args.model)
    config = load_inference_config(args.config)
    cases = load_prompts(args.prompts)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit 必须为正整数")
        cases = cases[: args.limit]
    adapter = args.adapter.expanduser().resolve() if args.adapter else None
    if adapter and not (
        (adapter / "adapter_config.json").is_file()
        and (adapter / "adapter_model.safetensors").is_file()
    ):
        parser.error(
            "--adapter 必须是含 adapter_config.json 和 adapter_model.safetensors 的本地 checkpoint"
        )
    kind = "lora" if adapter else "base"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / f"{kind}-{timestamp}"
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"输出目录已存在，拒绝覆盖：{output_dir}")
    print(f"模型：{model.name}\n本地路径：{model.path}\nAdapter：{adapter}", flush=True)
    print(f"提示词：{len(cases)} 条，SHA256={prompts_fingerprint(cases)}", flush=True)
    print(f"推理参数：{asdict(config)}\n输出目录：{output_dir}", flush=True)
    check_local_model(model)
    if args.dry_run:
        print("Dry run 通过。未加载模型、未创建输出目录。")
        return 0

    check_python()
    check_gpu()
    print("开始将本地 BF16 权重加载到 GPU……", flush=True)
    engine = create_engine(model, config, adapter)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "kind": kind,
        "model": {"name": model.name, "path": str(model.path), "model_type": model.model_type},
        "model_config_sha256": hashlib.sha256(
            (model.path / "config.json").read_bytes()
        ).hexdigest(),
        "adapter": str(adapter) if adapter else None,
        "prompts_file": str(args.prompts.resolve()),
        "prompts_sha256": prompts_fingerprint(cases),
        "prompt_ids": [case["id"] for case in cases],
        "inference": asdict(config),
        "packages": {name: version(name) for name in ("torch", "transformers", "ms-swift", "peft")},
    }
    results_path = write_run_results(engine, cases, config, output_dir, metadata)
    print(f"\n推理完成，结果保存在：{results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
