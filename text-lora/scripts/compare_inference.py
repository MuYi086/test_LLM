"""核对固定评估的模型/提示词/解码设置，按 ID 生成原始输出对照。"""

import argparse
import json
import re
from pathlib import Path

from prepare_training_data import TAG
from run_inference import prompts_fingerprint

EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*")


def format_label(category, body, metrics):
    if category == "general_control":
        return "出现演绎标签" if any(TAG.fullmatch(x) for x in body.splitlines()) else "普通文本"
    return "通过" if metrics["full_script_format"] else "未通过"


def body_and_format(record):
    original = record["output"]
    body = EMPTY_THINK.sub("", original, count=1)
    lines = body.splitlines()
    matches = [TAG.fullmatch(line) for line in lines]
    relaxed_lines = [re.sub(r"(^\[[^\[\]\n]+\]\[[^\[\]\n]+\])\s+", r"\1", line) for line in lines]
    return body, {
        "empty_think_prefix": body != original,
        "nonempty_think": "<think>" in body or "</think>" in body,
        "full_script_format": bool(lines) and all(matches),
        "script_tags_complete": bool(lines) and all(TAG.fullmatch(line) for line in relaxed_lines),
        "invalid_lines": [line for line, match in zip(lines, matches, strict=True) if not match],
        "finish_reason": record["finish_reason"],
    }


def load_run(directory):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ids = [record["id"] for record in records]
    if (
        metadata["status"] != "complete"
        or len(set(ids)) != len(ids)
        or len(ids) != metadata["completed_count"]
        or len(ids) != metadata["prompt_count"]
        or set(ids) != set(metadata["prompt_ids"])
    ):
        raise ValueError(f"{directory}: 运行未完整完成或 ID 不一致")
    cases = [
        {
            k: v
            for k, v in record.items()
            if k not in {"output", "reasoning_content", "finish_reason", "elapsed_seconds", "usage"}
        }
        for record in records
    ]
    # 原基线结果写入顺序可能与 metadata 不同，按 metadata 恢复哈希顺序。
    by_id = {case["id"]: case for case in cases}
    ordered = [by_id[identifier] for identifier in metadata["prompt_ids"]]
    if prompts_fingerprint(ordered) != metadata["prompts_sha256"]:
        raise ValueError(f"{directory}: 保存的输入与提示词哈希不匹配")
    return metadata, {record["id"]: record for record in records}


def compare(base_dir, lora_dir, output_dir):
    base_meta, base = load_run(base_dir)
    lora_meta, lora = load_run(lora_dir)
    if base_meta["kind"] != "base" or lora_meta["kind"] != "lora" or not lora_meta["adapter"]:
        raise ValueError("需要一组完整 base 和一组实际加载 adapter 的 lora 结果")
    for key in ("model", "model_config_sha256", "prompts_sha256", "inference", "packages"):
        if base_meta[key] != lora_meta[key]:
            raise ValueError(f"两组运行的 {key} 不一致，不生成直接对照")
    if set(base) != set(lora):
        raise ValueError("测试 ID 集合不同")
    rows, report = (
        [],
        [
            "# 固定 10 条：原模型 / LoRA 对照",
            "",
            f"原模型：`{base_dir}`",
            f"LoRA：`{lora_dir}`",
            f"Adapter：`{lora_meta['adapter']}`",
            "",
            "提示词、模型配置、依赖版本和解码参数一致；按 ID 对齐。",
            "仅去除开头的空 think 块用于正文格式检查；原始记录保持不变。",
            "严格格式还要求正文首字符非空白；metrics.json 另记录允许标签后空格的双标签检查。",
            "自动指标只检查标签格式和结束原因；事实、台词归属与新增内容需要逐条阅读。",
            "",
            "| ID | 原模型格式 | LoRA 格式 | 原模型/LoRA 空 think | 原模型/LoRA 结束 |",
            "| --- | --- | --- | --- | --- |",
        ],
    )
    for identifier in base_meta["prompt_ids"]:
        left, right = base[identifier], lora[identifier]
        if left["messages"] != right["messages"] or left["category"] != right["category"]:
            raise ValueError(f"{identifier}: 实际输入或任务类型发生变化")
        left_body, left_metrics = body_and_format(left)
        right_body, right_metrics = body_and_format(right)
        row = {
            "id": identifier,
            "category": left["category"],
            "base": left_metrics,
            "lora": right_metrics,
        }
        rows.append(row)

        report.append(
            f"| {identifier} | {format_label(left['category'], left_body, left_metrics)} | "
            f"{format_label(right['category'], right_body, right_metrics)} | "
            f"{left_metrics['empty_think_prefix']}/{right_metrics['empty_think_prefix']} | "
            f"{left_metrics['finish_reason']}/{right_metrics['finish_reason']} |"
        )
    for identifier in base_meta["prompt_ids"]:
        left, right = base[identifier], lora[identifier]
        report.extend(
            [
                "",
                f"## {identifier}",
                "",
                "输入：",
                "",
                left["messages"][-1]["content"],
                "",
                "原模型原始输出：",
                "",
                "```text",
                left["output"],
                "```",
                "",
                "LoRA 原始输出：",
                "",
                "```text",
                right["output"],
                "```",
                "",
                "检查项：",
                "",
            ]
        )
        report.extend(f"- {criterion}" for criterion in left.get("review_criteria", []))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = compare(args.base.resolve(), args.lora.resolve(), args.output_dir.resolve())
    for kind in ("base", "lora"):
        rewritten = [row[kind] for row in rows if row["category"] == "tts_rewrite"]
        print(
            f"{kind}: 正文完整标签格式 {sum(x['full_script_format'] for x in rewritten)}/{len(rewritten)}"
        )
        print(
            f"{kind}: 双标签与非空正文（允许标签后空格） {sum(x['script_tags_complete'] for x in rewritten)}/{len(rewritten)}"
        )
    print(f"[PASS] 对照已保存：{args.output_dir.resolve() / 'comparison.md'}")


if __name__ == "__main__":
    main()
