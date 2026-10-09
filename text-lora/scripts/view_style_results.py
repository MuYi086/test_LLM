"""独立只读查看风格样本、流程状态和对照；不启动训练或加载权重。"""

import argparse
import html
import json
import os
from pathlib import Path

from check_environment import PROJECT_ROOT


def read_status(project_root=PROJECT_ROOT):
    paths = list((project_root / "outputs").glob("suspense-*/style_metadata.json"))
    if not paths:
        return {"status": "not_started"}
    entries = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    path, metadata = max(entries, key=lambda pair: pair[1]["created_at"])
    info = {**metadata, "run_dir": str(path.parent)}
    if info["status"] == "running" and info.get("controller_pid"):
        try:
            os.kill(info["controller_pid"], 0)
        except ProcessLookupError:
            info["status"] = "interrupted"
    return info


def preview_samples(project_root=PROJECT_ROOT):
    from IPython.display import HTML, Markdown, display

    corpus = json.loads((project_root / "datasets/suspense_samples_v1.json").read_text())
    display(Markdown((project_root / "datasets/suspense_style_guide.md").read_text()))
    checks_path = project_root / "datasets/processed/teaching-v3/checks.json"
    if checks_path.is_file():
        from prepare_training_data import canonical_hash

        checks = json.loads(checks_path.read_text())
        if checks["corpus_sha256"] != canonical_hash(corpus["samples"]):
            raise ValueError("语料修改后尚未重新检查，请交给 agent 处理")
        maximum = max(row["total_tokens"] for row in checks["lengths"])
        print(f"[PASS] 已有 CPU 检查通过；30 条训练、10 条验证；完整模板最长 {maximum} tokens。")
    else:
        print("[STATUS] 样稿已准备；CPU 检查尚未保存，由步骤 2 自动完成。")
    blocks = []
    for case in corpus["samples"]:
        title = f"{case['split']} / {case['id']} / {case['title']}"
        blocks.append(
            "<details><summary>"
            + html.escape(title)
            + "</summary><p>原文：</p><pre>"
            + html.escape(case["source_text"])
            + "</pre><p>悬疑演绎参考：</p><pre>"
            + html.escape(case["messages"][-1]["content"])
            + "</pre></details>"
        )
    display(HTML("".join(blocks)))
    print("[VIEW] 样稿由 agent 编写和核对；风格是否采用仍由你决定。")


def display_results(project_root=PROJECT_ROOT):
    from IPython.display import Markdown, display

    print("[VIEW] 正在只读查看悬疑风格结果。", flush=True)
    info = read_status(project_root)
    if info["status"] != "complete":
        status = {
            "not_started": "尚未启动",
            "running": "正在执行",
            "failed": "执行失败",
            "interrupted": "执行进程已退出",
        }.get(info["status"], info["status"])
        print(
            f"[STATUS] 风格实验{status}；阶段：{info.get('stage_number', '-')}/8 {info.get('stage_title', '')}。",
            flush=True,
        )
        if info.get("error"):
            print(info["error"])
        print("步骤 2 包含训练和后续评估；等待最终 [COMPLETE] 和 Cell 的 [*] 消失。")
        return info
    root = Path(info["run_dir"])
    print(f"[COMPLETE] 8/8 阶段已完成：{root}", flush=True)
    display(Markdown((root / "summary.md").read_text(encoding="utf-8")))
    review = root / "agent_review.md"
    if review.is_file():
        display(Markdown(review.read_text(encoding="utf-8")))
    display(Markdown((root / "comparison/comparison.md").read_text(encoding="utf-8")))
    print(f"[VIEW] 朗读候选正文与角色/语气清单：{root / 'tts'}", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.preview:
        preview_samples()
    else:
        print(json.dumps(read_status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
