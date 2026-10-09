"""一键执行第二轮数据检查、LoRA 训练与原模型/两轮 adapter 对照。"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from check_environment import PROJECT_ROOT, load_model_config
from compare_inference import load_run
from prepare_training_data import (
    ROUND2_BENCHMARK,
    ROUND2_PROMPTS,
    canonical_hash,
    check_exported_data,
    export_benchmark,
    export_data,
    load_training_config,
    validate_corpus,
)
from run_inference import load_inference_config, prompts_fingerprint
from train_lora import verify_adapter
from view_second_iteration import stage_title

CORPUS_V2 = PROJECT_ROOT / "datasets/teaching_samples_v2.json"
LATEST = PROJECT_ROOT / "outputs/latest_iteration2.json"
LOCK = PROJECT_ROOT / "outputs/iteration2.lock"


def handle_stop(signum, frame):
    raise KeyboardInterrupt("第二轮执行已中断")


@contextmanager
def iteration_lock():
    """阻止两个 Notebook/kernel 同时启动同一条单卡实验流程。"""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def progress_message(label, line):
    line = line.strip()
    title = stage_title(label)
    match = re.match(r"^\[(\d+)/(\d+)\]\s+(.+)$", line)
    if match:
        current, total, identifier = match.groups()
        return f"[PROGRESS] [{title}] 正在推理第 {current}/{total} 条：{identifier}"
    if "[PASS] 训练完成" in line:
        next_action = "随后继续正式训练" if label == "train-smoke" else "随后继续推理与报告"
        return f"[PASS] [{title}] 当前训练阶段通过；{next_action}。全流程结束以 [COMPLETE] 为准。"
    if any(tag in line for tag in ("{'loss':", "{'eval_loss':")):
        return f"[PROGRESS] [{title}] {line}"
    if "[PASS]" in line or "推理完成" in line:
        return f"[PASS] [{title}] {line.removeprefix('[PASS]').strip()}"
    return None


def current_state(corpus):
    fresh = export_benchmark()
    samples, fixed = validate_corpus(corpus)
    export_data(samples, corpus)
    check_exported_data(samples, corpus)
    first_receipt = json.loads((PROJECT_ROOT / "outputs/latest_teaching_run.json").read_text())
    first_dir = PROJECT_ROOT / first_receipt["full_dir"]
    first = json.loads((first_dir / "run_metadata.json").read_text())
    if first["status"] != "complete" or first["mode"] != "full":
        raise ValueError("第一轮正式训练尚未完成")
    first_adapter = Path(first["selected_adapter"]["path"])
    verify_adapter(first_adapter)
    base_dir = PROJECT_ROOT / "outputs/base-20261008T012651_918242Z"
    first_fixed = PROJECT_ROOT / first_receipt["lora_dir"]
    base_meta, _ = load_run(base_dir)
    first_meta, _ = load_run(first_fixed)
    model = load_model_config()
    packages = {name: version(name) for name in ("torch", "transformers", "ms-swift", "peft")}
    model_hash = hashlib.sha256((model.path / "config.json").read_bytes()).hexdigest()
    inference = asdict(load_inference_config())
    for metadata in (base_meta, first_meta):
        if (
            metadata["model_config_sha256"] != model_hash
            or metadata["packages"] != packages
            or metadata["inference"] != inference
            or metadata["prompts_sha256"] != prompts_fingerprint(fixed)
            or metadata["model"]["path"] != str(model.path)
        ):
            raise ValueError("当前环境、模型或固定评估配置与第一轮不一致")
    if Path(first_meta["adapter"]).resolve() != first_adapter.resolve():
        raise ValueError("第一轮固定推理与选中的 adapter 不一致")
    return {
        "corpus_file": str(corpus.resolve()),
        "corpus_sha256": canonical_hash(samples),
        "benchmark_sha256": hashlib.sha256(ROUND2_BENCHMARK.read_bytes()).hexdigest(),
        "fresh_prompts_sha256": prompts_fingerprint(fresh),
        "fixed_prompts_sha256": prompts_fingerprint(fixed),
        "training": load_training_config(),
        "inference": inference,
        "packages": packages,
        "model_config_sha256": model_hash,
        "first_adapter": str(first_adapter),
        "base_fixed_dir": str(base_dir),
        "first_fixed_dir": str(first_fixed),
    }


def run_stage(root, label, script, *arguments):
    command = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / script), *map(str, arguments)]
    log_path = root / f"{label}.log"
    print(f"[RUN] [STAGE {stage_title(label)}]；完整日志：{log_path}", flush=True)
    process = subprocess.Popen(
        command,
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
                message = progress_message(label, line)
                if message:
                    print(message, flush=True)
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
        raise RuntimeError(f"{label} 失败，exit={code}")
    print(f"[PASS] [STAGE-DONE {stage_title(label)}] 本阶段完成。", flush=True)


def reuse_completed(state):
    if not LATEST.exists():
        return None
    receipt = json.loads(LATEST.read_text())
    root = Path(receipt["run_dir"])
    metadata = json.loads((root / "iteration_metadata.json").read_text())
    if metadata["status"] != "complete" or metadata["state"] != state:
        return None
    verify_adapter(metadata["second_adapter"])
    for key in ("second_fixed_dir", "base_fresh_dir", "first_fresh_dir", "second_fresh_dir"):
        load_run(Path(metadata[key]))
    for key in ("fixed_comparison_dir", "fresh_comparison_dir"):
        if not (Path(metadata[key]) / "comparison.md").is_file():
            raise ValueError("已完成记录缺少对照报告")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_V2)
    parser.add_argument("--dry-run", action="store_true", help="核对计划，不启动 GPU 训练或推理")
    parser.add_argument("--new-run", action="store_true", help="忽略已完成的匹配结果，重新实验")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, handle_stop)
    args.corpus = args.corpus.expanduser().resolve()
    state = current_state(args.corpus)
    if args.dry_run:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        print(
            "[PASS] 计划：CPU模板检查 → 3步试跑 → 3轮全新LoRA → 原10条第二轮推理 → "
            "新10条原模型/第一轮/第二轮推理 → 两份三组对照。未启动GPU任务。"
        )
        return 0
    with iteration_lock() as acquired:
        if not acquired:
            print(
                "[STATUS] 第二轮已有执行任务；本次调用未启动新任务。步骤 3 可查看最新状态。",
                flush=True,
            )
            return 0
        return execute_iteration(args.corpus, state, args.new_run)


def execute_iteration(corpus, state, new_run=False):
    if not new_run:
        existing = reuse_completed(state)
        if existing:
            print(
                f"[PASS] [COMPLETE] 第二轮 9/9 阶段已完成，复用结果：{existing}。直接运行步骤 3。"
            )
            return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    root = PROJECT_ROOT / "outputs" / f"iteration2-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "state": state,
        "stage": "checks",
        "controller_pid": os.getpid(),
    }

    def save():
        path = root / "iteration_metadata.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    save()
    try:
        run_stage(root, "checks", "prepare_training_data.py", "--corpus", corpus, "--check-tokens")
        for mode in ("smoke", "full"):
            metadata["stage"] = "train-" + mode
            save()
            run_stage(
                root,
                "train-" + mode,
                "train_lora.py",
                "--corpus",
                corpus,
                "--mode",
                mode,
                "--output-dir",
                root / ("train-" + mode),
            )
        full = json.loads((root / "train-full/run_metadata.json").read_text())
        adapter = Path(full["selected_adapter"]["path"])
        metadata["second_adapter"] = str(adapter)
        jobs = [
            ("second-fixed", PROJECT_ROOT / "eval/prompts.jsonl", adapter),
            ("base-fresh", ROUND2_PROMPTS, None),
            ("first-fresh", ROUND2_PROMPTS, Path(state["first_adapter"])),
            ("second-fresh", ROUND2_PROMPTS, adapter),
        ]
        for label, prompts, selected in jobs:
            metadata["stage"] = label
            save()
            extra = ["--adapter", str(selected)] if selected else []
            run_stage(
                root,
                label,
                "run_inference.py",
                "--prompts",
                prompts,
                "--output-dir",
                root / label,
                *extra,
            )
            metadata[label.replace("-", "_") + "_dir"] = str(root / label)
        for name, base, first, second in [
            (
                "fixed",
                Path(state["base_fixed_dir"]),
                Path(state["first_fixed_dir"]),
                root / "second-fixed",
            ),
            ("fresh", root / "base-fresh", root / "first-fresh", root / "second-fresh"),
        ]:
            metadata["stage"] = "compare-" + name
            save()
            extra = ["--references", ROUND2_BENCHMARK] if name == "fresh" else []
            run_stage(
                root,
                "compare-" + name,
                "compare_iterations.py",
                "--base",
                base,
                "--first",
                first,
                "--second",
                second,
                "--output-dir",
                root / ("comparison-" + name),
                *extra,
            )
            metadata[name + "_comparison_dir"] = str(root / ("comparison-" + name))
        metadata.update(
            status="complete", stage="complete", finished_at=datetime.now(UTC).isoformat()
        )
        save()
        temporary_receipt = LATEST.with_suffix(".tmp")
        temporary_receipt.write_text(
            json.dumps({"run_dir": str(root)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_receipt.replace(LATEST)
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    print(f"[PASS] [COMPLETE] 第二轮 9/9 阶段全部完成：{root}。现在可运行步骤 3 查看两份对照。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
