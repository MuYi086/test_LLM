"""导出风格候选的朗读正文和演绎标记；不生成音频、不改原始推理记录。"""

import argparse
import json
from collections import Counter
from pathlib import Path

from compare_inference import body_and_format, load_run
from prepare_style_data import BENCHMARK, TONES
from prepare_training_data import TAG


def export_results(root, benchmark_path=BENCHMARK):
    root = Path(root)
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    references = {case["id"]: case for case in benchmark["cases"]}
    metadata, records = load_run(root / "style")
    directory = root / "tts"
    directory.mkdir(exist_ok=False)
    manifest, counts = [], Counter()
    for identifier in metadata["prompt_ids"]:
        record, case = records[identifier], references[identifier]
        if record["messages"] != case["messages"]:
            raise ValueError("朗读导出与当前参考集的输入不一致")
        if case["category"] != "tts_rewrite":
            continue
        body, metrics = body_and_format(record)
        lines = [TAG.fullmatch(line.strip()) for line in body.splitlines()]
        valid = (
            bool(lines)
            and all(lines)
            and not metrics["nonempty_think"]
            and record["finish_reason"] == "stop"
        )
        if valid:
            valid = all(
                line.group(1) in case["allowed_roles"] and line.group(2) in TONES for line in lines
            )
        entry = {
            "id": identifier,
            "title": case["title"],
            "format_complete": bool(valid),
            "finish_reason": record["finish_reason"],
            "approved_for_audio": False,
            "note": "仍需核对语义与氛围；标记未映射到具体 TTS 参数",
        }
        if valid:
            case_dir = directory / identifier
            case_dir.mkdir()
            performance = [
                {"speaker": line.group(1), "tone": line.group(2), "text": line.group(3)}
                for line in lines
            ]
            counts.update(row["tone"] for row in performance)
            (case_dir / "performance.json").write_text(
                json.dumps(performance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (case_dir / "speech.txt").write_text(
                "\n".join(row["text"] for row in performance) + "\n", encoding="utf-8"
            )
            entry.update(
                speech_file=str(case_dir / "speech.txt"),
                performance_file=str(case_dir / "performance.json"),
            )
        manifest.append(entry)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = json.loads((root / "comparison/metrics.json").read_text())
    stories = [m for m in metrics if m["category"] == "tts_rewrite"]
    full = json.loads((root / "train-full/run_metadata.json").read_text())
    lines = [
        "# 悬疑风格实验：待监督的候选",
        "",
        f"运行目录：`{root}`",
        "",
        f"正式训练：{full['result']['global_step']} 个优化步；按独立验证 loss 选择 checkpoint。",
        "",
        "目标为适度改写措辞、断句和语气，保留人物事实，不添加情节。三组使用同一份新风格指令与解码设置。",
        "",
        "| 模型 | 改写正文双标签 | 输出触及上限 |",
        "| --- | --- | --- |",
    ]
    for key, label in (
        ("base", "原模型"),
        ("first", "第二轮保守 LoRA"),
        ("second", "悬疑风格 LoRA"),
    ):
        lines.append(
            f"| {label} | {sum(m[key]['script_tags_complete'] for m in stories)}/{len(stories)} | {sum(m[key]['finish_reason'] == 'length' for m in metrics)}/{len(metrics)} |"
        )
    lines += [
        "",
        "上表只说明格式与输出长度。内容忠实度、氛围和朗读自然度需要逐条审阅，不能用标签数量证明改写有效。",
        "",
        f"可导出的完整演绎稿：{sum(e['format_complete'] for e in manifest)}/{len(manifest)}。",
        "",
        f"演绎标记分布：`{dict(counts)}`。这不是声音情绪评分。",
        "",
        "## 你接下来监督什么",
        "",
        "1. 优先看两个较长片段：有没有遗漏动作、数量、空间关系或冒号前的动作？",
        "2. 再看重复话语、纸面文字、内心可能性和普通任务：是否保留原意、没有串角色？",
        "3. 查看平稳段落是否仍然克制，按你的 TTS 试听后决定改写强度。",
        "",
        "朗读导出在 `tts/`：`speech.txt` 不含演绎标签；`performance.json` 保留角色、语气与正文，等待接入具体 TTS。尚未生成音频，也没有代替用户批准这些候选。",
        "",
        "完整对照见 `comparison/comparison.md`；原始输出继续保存在 `base/`、`neutral/`、`style/`。",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[PASS] 已导出候选朗读正文、演绎清单与监督摘要：{root / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    args = parser.parse_args()
    export_results(args.run_dir, args.benchmark)


if __name__ == "__main__":
    main()
