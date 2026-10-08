"""从本地原模型启动单卡文本 LoRA，保存训练状态并核验 adapter。"""

import argparse
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from check_environment import (
    PROJECT_ROOT,
    check_gpu,
    check_local_model,
    check_python,
    load_model_config,
)
from prepare_training_data import (
    CORPUS,
    canonical_hash,
    check_exported_data,
    dataset_paths,
    load_training_config,
    validate_corpus,
)
from run_inference import prompts_fingerprint


def training_kwargs(mode, output_dir, corpus_path=CORPUS):
    model, config = load_model_config(), load_training_config()
    paths = dataset_paths(corpus_path)
    is_smoke = mode == "smoke"
    return dict(
        model=str(model.path),
        model_type=model.model_type,
        template=model.model_type,
        model_kwargs={"local_files_only": True},
        use_hf=True,
        torch_dtype="bfloat16",
        bf16=True,
        attn_impl="sdpa",
        device_map="cuda:0",
        tuner_type="lora",
        tuner_backend="peft",
        target_modules=["all-linear"],
        freeze_llm=False,
        freeze_vit=True,
        freeze_aligner=True,
        lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        dataset=[str(paths["train"])],
        val_dataset=[str(paths["val"])],
        split_dataset_ratio=0.0,
        max_length=config["max_length"],
        truncation_strategy="raise",
        strict=True,
        enable_thinking=False,
        loss_scale=config["loss_scale"],
        packing=False,
        padding_free=False,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        load_from_cache_file=False,
        dataset_shuffle=True,
        train_dataloader_shuffle=True,
        gradient_checkpointing=True,
        vit_gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["smoke_steps"] if is_smoke else -1,
        learning_rate=config["learning_rate"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        seed=config["seed"],
        data_seed=config["data_seed"],
        logging_steps=1,
        logging_nan_inf_filter=False,
        eval_strategy="steps" if is_smoke else "epoch",
        save_strategy="steps" if is_smoke else "epoch",
        eval_steps=config["smoke_steps"] if is_smoke else None,
        save_steps=config["smoke_steps"] if is_smoke else 500,
        save_total_limit=2,
        save_only_model=False,
        load_best_model_at_end=not is_smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        output_dir=str(output_dir),
        add_version=False,
        report_to=["none"],
        push_to_hub=False,
    )


def verify_adapter(path):
    path = Path(path)
    for filename in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
        if not (path / filename).is_file() or (path / filename).stat().st_size == 0:
            raise ValueError(f"checkpoint 文件缺失或为空：{path / filename}")
    from safetensors import safe_open

    with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as file:
        keys = list(file.keys())
        if not keys or any("lora_" not in key or "visual" in key for key in keys):
            raise ValueError("adapter 包含非 LoRA 参数或视觉参数")
    return {
        "path": str(path.resolve()),
        "tensor_count": len(keys),
        "weights_bytes": (path / "adapter_model.safetensors").stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--dry-run", action="store_true", help="只核对数据和已安装参数字段")
    args = parser.parse_args()
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "USE_HF"):
        os.environ[key] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    samples, prompts = validate_corpus(args.corpus)
    check_exported_data(samples, args.corpus)
    paths = dataset_paths(args.corpus)
    model = load_model_config()
    check_python()
    check_local_model(model)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or PROJECT_ROOT / f"outputs/train-{args.mode}-{stamp}").resolve()
    if output.exists():
        parser.error(f"拒绝覆盖已有运行目录：{output}")
    kwargs = training_kwargs(args.mode, output, args.corpus)
    from dataclasses import fields

    from swift.arguments import SftArguments

    unknown = set(kwargs) - {field.name for field in fields(SftArguments)}
    if unknown:
        raise ValueError(f"已安装 ms-swift 不支持参数：{sorted(unknown)}")
    print(json.dumps(kwargs, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("[PASS] dry run；未实例化 SftArguments、未加载权重、未创建运行目录。")
        return 0
    check_gpu()
    import torch
    from swift.pipelines.train.sft import SwiftSft
    from transformers import TrainerCallback

    class FiniteLoss(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "eval_loss", "grad_norm"):
                if key in (logs or {}) and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"step={state.global_step}: {key}={logs[key]}")

    class TextOnlySft(SwiftSft):
        @classmethod
        def prepare_model(cls, args, model, **kwargs):
            model = super().prepare_model(args, model, **kwargs)
            names = [n for n, p in model.named_parameters() if p.requires_grad]
            if not names or any("lora_" not in n or "visual" in n for n in names):
                raise ValueError("发现非 LoRA 或视觉部分可训练参数")
            (output / "trainable_parameters.json").write_text(
                json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[PASS] {len(names)} 个可训练张量，均为文本 LoRA。", flush=True)
            return model

        def train(self, trainer):
            trainer.add_callback(FiniteLoss())
            return super().train(trainer)

    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        (output / f"{split}.jsonl").write_bytes(paths[split].read_bytes())
    # 训练使用运行内快照，避免 Notebook 修改数据影响正在执行的任务。
    kwargs["dataset"] = [str(output / "train.jsonl")]
    kwargs["val_dataset"] = [str(output / "val.jsonl")]
    metadata = {
        "schema_version": 1,
        "kind": "lora_training",
        "mode": args.mode,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_sha256": canonical_hash(samples),
        "corpus_file": str(args.corpus.resolve()),
        "prompts_sha256": prompts_fingerprint(prompts),
        "model_config_sha256": hashlib.sha256(
            (model.path / "config.json").read_bytes()
        ).hexdigest(),
        "kwargs": kwargs,
        "packages": {name: version(name) for name in ("torch", "transformers", "ms-swift", "peft")},
    }

    def save_metadata():
        (output / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    save_metadata()
    try:
        result = TextOnlySft(SftArguments(**kwargs)).main()
        if not result["global_step"] or not any("loss" in x for x in result["log_history"]):
            raise ValueError("训练没有完成有效优化步数")
        checkpoint = result.get("best_model_checkpoint") or result.get("last_model_checkpoint")
        adapter = verify_adapter(checkpoint)
        metadata.update(status="complete", result=result, selected_adapter=adapter)
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata.update(
            elapsed_seconds=round(time.perf_counter() - started, 3),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            finished_at=datetime.now(UTC).isoformat(),
        )
        save_metadata()
    print(f"[PASS] 训练完成：{output}\n用于推理的 adapter：{adapter['path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
