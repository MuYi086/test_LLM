"""只读显示第二轮状态和已有对照；不导入训练入口或启动子进程。"""

import argparse
import json
import os
from pathlib import Path

from check_environment import PROJECT_ROOT

STAGES = (
    ("checks", "数据与模板检查"),
    ("train-smoke", "3 步试跑"),
    ("train-full", "3 轮正式训练"),
    ("second-fixed", "原 10 条：第二轮推理"),
    ("base-fresh", "新 10 条：原模型推理"),
    ("first-fresh", "新 10 条：第一轮 LoRA 推理"),
    ("second-fresh", "新 10 条：第二轮 LoRA 推理"),
    ("compare-fixed", "生成原测试对照"),
    ("compare-fresh", "生成新测试对照"),
)


def stage_title(label):
    for index, (key, title) in enumerate(STAGES, 1):
        if key == label:
            return f"{index}/{len(STAGES)} {title}"
    return "9/9 全流程完成" if label == "complete" else str(label)


def read_iteration(project_root=PROJECT_ROOT, run_dir=None):
    """即使全流程未完成，也显示最新阶段，不将训练完成当成全流程完成。"""
    if run_dir is None:
        candidates = set((project_root / "outputs").glob("iteration2-*"))
        receipt_path = project_root / "outputs/latest_iteration2.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            candidates.add(Path(receipt["run_dir"]))
        available = []
        for candidate in candidates:
            path = candidate / "iteration_metadata.json"
            if path.is_file():
                metadata = json.loads(path.read_text(encoding="utf-8"))
                available.append((metadata.get("created_at", candidate.name), candidate, metadata))
        if not available:
            return {"status": "not_started", "stage": None, "run_dir": None}
        _, run_dir, metadata = max(available, key=lambda item: item[0])
    else:
        run_dir = Path(run_dir)
        metadata = json.loads((run_dir / "iteration_metadata.json").read_text(encoding="utf-8"))
    status = metadata["status"]
    if status == "running" and metadata.get("controller_pid"):
        try:
            os.kill(metadata["controller_pid"], 0)
        except ProcessLookupError:
            status = "interrupted"
        except PermissionError:
            pass
    return {
        "status": status,
        "stage": metadata.get("stage"),
        "run_dir": str(run_dir),
        "metadata": metadata,
    }


def result_summary(info):
    status, root = info["status"], info["run_dir"]
    if status == "not_started":
        return "[STATUS] 第二轮尚未启动。采用方案后执行步骤 2 的代码 Cell。"
    if status != "complete":
        state = {"running": "正在执行", "failed": "执行失败", "interrupted": "执行进程已退出"}.get(
            status, status
        )
        error = info["metadata"].get("error", "")
        return (
            f"[STATUS] 第二轮{state}；当前阶段：{stage_title(info['stage'])}。\n\n"
            f"运行目录：`{root}`\n\n"
            "步骤 2 包含训练、推理和报告。等待最终 `[COMPLETE]` 和 Cell 的 `[*]` 消失后查看完整结果。"
            + (f"\n\n错误：`{error}`" if error else "")
        )
    metadata = info["metadata"]
    training = json.loads((Path(root) / "train-full/run_metadata.json").read_text(encoding="utf-8"))
    if training["status"] != "complete":
        raise ValueError("全流程记录与训练状态不一致")
    lines = [
        "[COMPLETE] 第二轮 9/9 阶段全部完成，已有结果可直接查看。",
        "",
        f"运行目录：`{root}`",
        "",
        f"正式训练：{training['result']['global_step']} 个优化步；"
        f"PyTorch 峰值 allocated/reserved：{training['peak_allocated_gib']}/{training['peak_reserved_gib']} GiB。",
        "",
        "下表只检查正文双标签与非空内容，内容质量仍按逐条检查项评审。",
        "",
        "| 测试集合 | 原模型格式 | 第一轮格式 | 第二轮格式 |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in (("新 10 条", "fresh_comparison_dir"), ("原 10 条", "fixed_comparison_dir")):
        directory = Path(metadata[key])
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        cases = [case for case in metrics if case["category"] == "tts_rewrite"]
        values = [
            f"{sum(case[k]['script_tags_complete'] for case in cases)}/{len(cases)}"
            for k in ("base", "first", "second")
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
        if not (directory / "comparison.md").is_file():
            raise FileNotFoundError(directory / "comparison.md")
    lines.extend(
        [
            "",
            "逐条对照：",
            "",
            f"- 新测试：`{Path(metadata['fresh_comparison_dir']) / 'comparison.md'}`",
            f"- 原测试：`{Path(metadata['fixed_comparison_dir']) / 'comparison.md'}`",
        ]
    )
    return "\n".join(lines)


def display_results(project_root=PROJECT_ROOT, run_dir=None):
    from IPython.display import Markdown, display

    print("[VIEW] 步骤 3：正在只读加载已有状态与报告。", flush=True)
    info = read_iteration(project_root, run_dir)
    display(Markdown(result_summary(info)))
    if info["status"] == "complete":
        for title, key in (
            ("新 10 条未见测试", "fresh_comparison_dir"),
            ("原有 10 条测试", "fixed_comparison_dir"),
        ):
            display(Markdown("### " + title))
            report = Path(info["metadata"][key]) / "comparison.md"
            display(Markdown(report.read_text(encoding="utf-8")))
        print("[VIEW] 两份逐条对照已显示。", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    info = read_iteration(run_dir=args.run_dir)
    print(json.dumps(info, ensure_ascii=False, indent=2) if args.json else result_summary(info))


if __name__ == "__main__":
    main()
