"""自动完成悬疑风格实验：检查、训练、三组对照及供试听的文本导出。"""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from check_environment import PROJECT_ROOT, load_model_config
from compare_inference import load_run
from prepare_style_data import CORPUS, evaluation_paths, validate_style_corpus
from prepare_training_data import canonical_hash, load_training_config
from run_inference import load_inference_config, prompts_fingerprint
from run_second_iteration import iteration_lock
from train_lora import verify_adapter

CONFIG = PROJECT_ROOT / "configs/style_inference.toml"
LATEST = PROJECT_ROOT / "outputs/latest_suspense_run.json"
STAGES = (
    ("checks", "数据与 CPU 模板检查"),
    ("train-smoke", "3 步试跑"),
    ("train-full", "3 轮悬疑风格训练"),
    ("base", "原模型的风格测试"),
    ("neutral", "第二轮保守 LoRA 的风格测试"),
    ("style", "悬疑风格 LoRA 的风格测试"),
    ("compare", "生成三组内容对照"),
    ("export", "导出无标签朗读稿与演绎清单"),
)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def current_state(corpus_path=CORPUS):
    samples, prompts = validate_style_corpus(corpus_path)
    benchmark_path, _ = evaluation_paths(corpus_path)
    receipt = json.loads((PROJECT_ROOT / "outputs/latest_iteration2.json").read_text())
    neutral = json.loads((Path(receipt["run_dir"]) / "iteration_metadata.json").read_text())
    if neutral["status"] != "complete":
        raise ValueError("需要已完成的第二轮保守 LoRA")
    adapter = Path(neutral["second_adapter"])
    verify_adapter(adapter)
    model = load_model_config()
    return {
        "corpus_sha256": canonical_hash(samples),
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "prompts_sha256": prompts_fingerprint(prompts),
        "training": load_training_config(),
        "inference": asdict(load_inference_config(CONFIG)),
        "neutral_adapter": str(adapter),
        "neutral_weights_sha256": hashlib.sha256(
            (adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "model": str(model.path),
        "model_config_sha256": hashlib.sha256(
            (model.path / "config.json").read_bytes()
        ).hexdigest(),
        "packages": {name: version(name) for name in ("torch", "transformers", "ms-swift", "peft")},
    }


def read_status():
    candidates = list((PROJECT_ROOT / "outputs").glob("suspense-*/style_metadata.json"))
    if not candidates:
        return None
    values = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in candidates]
    path, metadata = max(values, key=lambda item: item[1]["created_at"])
    metadata = {**metadata, "run_dir": str(path.parent)}
    if metadata["status"] == "running" and metadata.get("controller_pid"):
        try:
            os.kill(metadata["controller_pid"], 0)
        except ProcessLookupError:
            metadata["status"] = "interrupted"
    return metadata


def completed(state, latest=LATEST):
    if not latest.exists():
        return None
    root = Path(json.loads(latest.read_text())["run_dir"])
    metadata = json.loads((root / "style_metadata.json").read_text())
    if metadata["status"] != "complete" or metadata["state"] != state:
        return None
    verify_adapter(metadata["style_adapter"])
    for key in ("base", "neutral", "style"):
        load_run(root / key)
    for name in ("comparison/comparison.md", "summary.md", "tts/manifest.json"):
        if not (root / name).is_file():
            raise ValueError(f"完整记录缺少文件：{name}")
    return root


def stage(root, index, script, *arguments):
    label, title = STAGES[index]
    prefix = f"{index + 1}/{len(STAGES)} {title}"
    log_path = root / (label + ".log")
    print(f"[RUN] [{prefix}]", flush=True)
    process = subprocess.Popen(
        [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / script), *map(str, arguments)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        with log_path.open("x", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                match = re.match(r"^\[(\d+)/(\d+)\]\s+(.+)$", line.strip())
                if match:
                    print(
                        f"[PROGRESS] [{prefix}] 第 {match[1]}/{match[2]} 条：{match[3]}", flush=True
                    )
                elif "[PASS] 训练完成" in line:
                    print(
                        f"[PASS] [{prefix}] 当前训练阶段通过；后续还有评估，以 [COMPLETE] 为准。",
                        flush=True,
                    )
                elif any(
                    tag in line for tag in ("[PASS]", "{'loss':", "{'eval_loss':", "推理完成")
                ):
                    print(f"[PROGRESS] [{prefix}] {line.strip()}", flush=True)
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
    if code:
        print(log_path.read_text(encoding="utf-8")[-12000:], flush=True)
        raise RuntimeError(f"{label} 失败；日志：{log_path}")
    print(f"[PASS] [{prefix}] 本阶段完成。", flush=True)


def execute(state, new_run=False, corpus_path=CORPUS, latest=LATEST, prefix="suspense"):
    benchmark_path, prompts_path = evaluation_paths(corpus_path)
    previous = None if new_run else completed(state, latest)
    if previous:
        print(f"[COMPLETE] 8/8 阶段已完成，复用已有悬疑风格结果：{previous}", flush=True)
        return previous
    root = (
        PROJECT_ROOT / "outputs" / (prefix + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ"))
    )
    root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "controller_pid": os.getpid(),
        "state": state,
        "stage": "checks",
        "corpus_file": str(corpus_path.resolve()),
        "benchmark_file": str(benchmark_path.resolve()),
    }
    path = root / "style_metadata.json"
    atomic_json(path, metadata)
    try:
        for index, (label, title) in enumerate(STAGES):
            metadata.update(stage=label, stage_number=index + 1, stage_title=title)
            atomic_json(path, metadata)
            if label == "checks":
                stage(
                    root, index, "prepare_style_data.py", "--corpus", corpus_path, "--check-tokens"
                )
            elif label.startswith("train-"):
                stage(
                    root,
                    index,
                    "train_lora.py",
                    "--corpus",
                    corpus_path,
                    "--mode",
                    label.removeprefix("train-"),
                    "--output-dir",
                    root / label,
                )
                if label == "train-full":
                    full = json.loads((root / label / "run_metadata.json").read_text())
                    metadata["style_adapter"] = full["selected_adapter"]["path"]
            elif label in ("base", "neutral", "style"):
                adapter = {
                    "base": None,
                    "neutral": state["neutral_adapter"],
                    "style": metadata["style_adapter"],
                }[label]
                extra = ["--adapter", adapter] if adapter else []
                stage(
                    root,
                    index,
                    "run_inference.py",
                    "--prompts",
                    prompts_path,
                    "--config",
                    CONFIG,
                    "--output-dir",
                    root / label,
                    *extra,
                )
            elif label == "compare":
                stage(
                    root,
                    index,
                    "compare_iterations.py",
                    "--base",
                    root / "base",
                    "--first",
                    root / "neutral",
                    "--second",
                    root / "style",
                    "--output-dir",
                    root / "comparison",
                    "--references",
                    benchmark_path,
                    "--labels",
                    "原模型",
                    "第二轮保守 LoRA",
                    "悬疑风格 LoRA",
                )
            else:
                stage(
                    root,
                    index,
                    "export_style_results.py",
                    "--run-dir",
                    root,
                    "--benchmark",
                    benchmark_path,
                )
        metadata.update(
            status="complete", stage="complete", finished_at=datetime.now(UTC).isoformat()
        )
        atomic_json(path, metadata)
        atomic_json(latest, {"run_dir": str(root)})
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(path, metadata)
        raise
    print(f"[COMPLETE] 悬疑风格实验 8/8 阶段完成：{root}。现在查看步骤 3 的内容对照。", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--new-run", action="store_true", help="新建实验；默认复用完整且匹配的结果")
    parser.add_argument("--status", action="store_true", help="只读状态，不导入模型权重")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    args = parser.parse_args()
    if args.status and args.corpus.resolve() == CORPUS.resolve():
        print(json.dumps(read_status(), ensure_ascii=False, indent=2))
        return 0

    def stop(signum, frame):
        raise KeyboardInterrupt("悬疑风格流程已中断")

    signal.signal(signal.SIGTERM, stop)
    corpus_path = args.corpus.resolve()
    custom = corpus_path != CORPUS.resolve()
    latest = PROJECT_ROOT / "outputs/latest_voxcpm_style_run.json" if custom else LATEST
    if custom and args.status:
        print(latest.read_text() if latest.exists() else "尚未运行方案A训练")
        return 0
    state = current_state(corpus_path)
    if args.dry_run:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        print("[PASS] 8 阶段计划通过；未启动训练或加载权重。")
        return 0
    with iteration_lock() as acquired:
        if not acquired:
            print("[STATUS] 当前已有第二轮或风格实验在执行，本次没有启动新的 GPU 任务。")
            return 0
        execute(state, args.new_run, corpus_path, latest, "voxcpm-style" if custom else "suspense")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
