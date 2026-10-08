"""检查项目 kernel、训练依赖、CUDA 和本地模型；不安装依赖、不加载模型权重。"""

import argparse
import importlib
import json
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models.toml"
PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ms-swift": "swift",
    "transformers": "transformers",
    "peft": "peft",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "modelscope": "modelscope",
    "qwen-vl-utils": "qwen_vl_utils",
    "ipykernel": "ipykernel",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    path: Path
    model_type: str


def load_model_config(
    profile: str | None = None, config_path: Path = DEFAULT_CONFIG
) -> ModelConfig:
    """读取本地模型配置，不访问模型仓库。"""
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    profile = profile or config["default_model"]
    if profile not in config["models"]:
        raise ValueError(f"未知模型配置 {profile!r}，可选：{', '.join(config['models'])}")
    selected = config["models"][profile]
    path = Path(selected["path"]).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return ModelConfig(selected["name"], path.resolve(), selected["model_type"])


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def check_local_model(model: ModelConfig) -> None:
    """检查配置、聊天模板、tokenizer 和索引引用的权重文件存在且非空。"""
    print(f"模型：{model.name}\n路径：{model.path}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        file = model.path / name
        if not file.is_file() or file.stat().st_size == 0:
            raise FileNotFoundError(f"缺少文件或文件为空：{file}")

    config = read_json(model.path / "config.json")
    if config.get("model_type") != model.model_type:
        raise ValueError(
            f"模型类型不匹配：配置要求 {model.model_type}，实际为 {config.get('model_type')}"
        )
    tokenizer_config = read_json(model.path / "tokenizer_config.json")
    template_file = model.path / "chat_template.jinja"
    if not tokenizer_config.get("chat_template") and not (
        template_file.is_file() and template_file.stat().st_size > 0
    ):
        raise ValueError("本地 tokenizer 缺少 chat template")

    index_path = model.path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = read_json(index_path).get("weight_map", {})
        if not weight_map or not all(isinstance(name, str) for name in weight_map.values()):
            raise ValueError(f"权重索引为空或格式无效：{index_path}")
        names = sorted(set(weight_map.values()))
    else:
        names = ["model.safetensors"]

    total_bytes = 0
    for name in names:
        shard = (model.path / name).resolve()
        if not shard.is_relative_to(model.path):
            raise ValueError(f"权重索引中的路径超出模型目录：{name}")
        if not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(f"缺少权重或权重为空：{shard}")
        total_bytes += shard.stat().st_size
    print(
        f"架构：{config.get('architectures')}\n权重：{len(names)} 个文件，{total_bytes / 1024**3:.2f} GiB"
    )
    print("检查了文件存在性和非空状态；未进行权重校验和验证。")


def check_python() -> None:
    """确认 Python 3.12 和当前项目的独立虚拟环境。"""
    print(f"Python：{sys.version.split()[0]}\n解释器：{sys.executable}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("本项目配置为 Python 3.12，请核对 .python-version 和 kernel")
    expected_prefix = PROJECT_ROOT / ".venv"
    if Path(sys.prefix).resolve() != expected_prefix.resolve():
        raise RuntimeError(f"当前 Python 不属于 {expected_prefix}，请切换项目 kernel")


def check_packages() -> None:
    """列出实际版本并检查导入，集中报告缺失包或兼容性错误。"""
    errors = []
    for distribution, module in PACKAGES.items():
        try:
            installed_version = version(distribution)
            importlib.import_module(module)
            print(f"{distribution}: {installed_version}")
        except Exception as error:
            errors.append(f"{distribution}: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeError("依赖检查失败：\n" + "\n".join(errors))


def check_gpu() -> None:
    """确认 PyTorch CUDA、显卡可见性和 BF16 支持。"""
    import torch

    print(f"PyTorch：{torch.__version__}\nCUDA runtime：{torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False，请核对 CUDA wheel 和 WSL 驱动")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU / PyTorch 环境不支持 BF16")
    free, total = torch.cuda.mem_get_info()
    print(f"BF16：True\n显存：空闲 {free / 1024**3:.2f} / 总量 {total / 1024**3:.2f} GiB")


def check_tokenizer(model: ModelConfig) -> None:
    """离线验证 Transformers 架构识别和文本聊天模板，不加载权重。"""
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        str(model.path), local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model.path), local_files_only=True, trust_remote_code=False
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话介绍 LoRA。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not prompt:
        raise RuntimeError("聊天模板生成了空提示词")
    print(f"Transformers 已识别：{config.model_type}\nTokenizer：{type(tokenizer).__name__}")
    print(f"聊天模板可用：{len(prompt)} 个字符；未加载模型权重。")


def run_check(label: str, check: Callable[[], None]) -> bool:
    print(f"\n--- {label} ---")
    try:
        check()
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}")
        return False
    print("[PASS]")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="configs/models.toml 中的模型配置名称")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="自定义模型配置文件")
    parser.add_argument("--local-only", action="store_true", help="仅用标准库检查本地模型文件")
    args = parser.parse_args()
    try:
        model = load_model_config(args.model, args.config)
    except Exception as error:
        print(f"[FAIL] 模型配置：{type(error).__name__}: {error}")
        return 1

    results = [run_check("本地模型文件", lambda: check_local_model(model))]
    if not args.local_only:
        results.extend(
            [
                run_check("Python / kernel", check_python),
                run_check("训练依赖", check_packages),
                run_check("GPU / BF16", check_gpu),
                run_check("离线架构 / tokenizer", lambda: check_tokenizer(model)),
            ]
        )
    passed = all(results)
    print("\n所选检查全部通过。" if passed else "\n存在失败项，请处理后重新检查。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
