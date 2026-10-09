"""同一组输入下核对原模型、第一轮与第二轮 LoRA 的原始输出。"""

import argparse
import json
from pathlib import Path

from compare_inference import body_and_format, load_run


def compare_iterations(base_dir, first_dir, second_dir, output_dir, references=None, labels=None):
    runs = [load_run(path) for path in (base_dir, first_dir, second_dir)]
    metadata = [run[0] for run in runs]
    if [m["kind"] for m in metadata] != ["base", "lora", "lora"]:
        raise ValueError("需要原模型、第一轮 adapter、第二轮 adapter 三组完整结果")
    if Path(metadata[1]["adapter"]).resolve() == Path(metadata[2]["adapter"]).resolve():
        raise ValueError("两轮使用了同一个 adapter，不能作为迭代对照")
    for key in ("model", "model_config_sha256", "prompts_sha256", "inference", "packages"):
        if any(m[key] != metadata[0][key] for m in metadata[1:]):
            raise ValueError(f"三组运行的 {key} 不一致")
    ids = metadata[0]["prompt_ids"]
    if any(set(records) != set(ids) for _, records in runs):
        raise ValueError("三组测试 ID 不一致")
    reference_map = {}
    if references:
        reference_map = {case["id"]: case for case in json.loads(references.read_text())["cases"]}
        if set(reference_map) != set(ids):
            raise ValueError("参考集 ID 与实际测试不一致")
    labels = labels or ["原模型", "第一轮 LoRA", "第二轮 LoRA"]
    if len(labels) != 3 or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("需要三个非空模型名称")
    metrics, report = (
        [],
        [
            "# " + " / ".join(labels) + "输出对照",
            "",
            "三组模型配置、输入哈希、依赖版本和解码设置相同，按 ID 对齐。",
            "格式检查只去除开头空 think 块；原始输出仍全部保留。",
            "表中通过只代表双标签与非空正文（允许标签后空格）；内容须阅读原文和检查项。",
            "",
            "| ID | " + " | ".join(label + "格式" for label in labels) + " |",
            "| --- | --- | --- | --- |",
        ],
    )
    for identifier in ids:
        records = [run[1][identifier] for run in runs]
        if any(record["messages"] != records[0]["messages"] for record in records[1:]):
            raise ValueError(f"{identifier}: 实际输入不一致")
        if reference_map and reference_map[identifier]["messages"] != records[0]["messages"]:
            raise ValueError(f"{identifier}: 参考答案的来源输入不一致")
        values = [body_and_format(record)[1] for record in records]
        metrics.append(
            {
                "id": identifier,
                "category": records[0]["category"],
                **dict(zip(["base", "first", "second"], values, strict=True)),
            }
        )
        if records[0]["category"] == "tts_rewrite":
            formats = ["通过" if value["script_tags_complete"] else "未通过" for value in values]
        else:
            formats = ["见普通任务输出"] * 3
        report.append(f"| {identifier} | " + " | ".join(formats) + " |")
    for identifier in ids:
        records = [run[1][identifier] for run in runs]
        report.extend(
            ["", f"## {identifier}", "", "原文/任务：", "", records[0]["messages"][-1]["content"]]
        )
        for label, record in zip(labels, records, strict=True):
            report.extend(["", label + "原始输出：", "", "```text", record["output"], "```"])
        if identifier in reference_map:
            report.extend(
                [
                    "",
                    "训练前冻结的参考写法（其他表达可能同样合理）：",
                    "",
                    "```text",
                    reference_map[identifier]["reference_output"],
                    "```",
                ]
            )
        report.extend(["", "检查项：", ""])
        report.extend("- " + criterion for criterion in records[0].get("review_criteria", []))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[PASS] 三组对照已保存：{output_dir / 'comparison.md'}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "first", "second", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--labels", nargs=3, help="三组显示名称；默认保持原有两轮名称")
    args = parser.parse_args()
    compare_iterations(
        args.base, args.first, args.second, args.output_dir, args.references, args.labels
    )


if __name__ == "__main__":
    main()
