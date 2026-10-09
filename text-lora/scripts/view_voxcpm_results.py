"""只读显示方案 A 的试听和训练记录，不加载模型、不启动服务。"""

import json
from pathlib import Path

from check_environment import PROJECT_ROOT


def latest(name):
    """解析已完成运行的指针，未完成时不猜测输出路径。"""
    pointer = PROJECT_ROOT / "outputs" / name
    if not pointer.exists():
        return None
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"])


def show_status():
    """集中显示已完成的工作与仍待监督的项目。"""
    root = latest("latest_voxcpm_audition.json")
    if root is None:
        raise FileNotFoundError("尚无完整试听，请由 agent 完成生成后再打开")
    manifest = json.loads((root / "manifest.json").read_text())
    print(f"试听：{manifest['status']}，{len(manifest['entries'])}份；运行目录：{root}")
    if manifest.get("revision", 1) >= 6:
        print(f"已处理的最新意见：{manifest['feedback']['decision']['comments']}")
        print("本轮5份主要试听：1、2加强局部情绪，3、4收敛平静与对白，5为纯平静对照。")
        print("已认可的语速处理和角色接话目标保留；旧版及只改响度对照折叠显示。")
        print("采用你保存的“更明显”，按人物和语气分别控制；本轮未重训文本LoRA。")
    elif manifest.get("revision", 1) >= 5:
        decision = manifest["feedback"]["decision"]
        counts = manifest["audition_summary"]
        print(f"已处理的最新意见：{decision['comments']}")
        print(
            f"本轮{counts['current_versions']}份主要试听，其中{counts['new_examples']}份新例子；发声速度放缓约6%。"
        )
        print("先听编号1确认速度，再听新例子；旧语速参考折叠显示。")
        print("新稿由agent编写审阅，用于声音控制试听；本轮不是新的LoRA预测或重训。")
    elif manifest.get("revision", 1) >= 4:
        decision = manifest["feedback"]["decision"]
        print(f"已处理的最新意见：{decision['comments']}")
        print("优先听两份“本轮”音频：旁白细微起伏、日常对白与角色音量。")
        print("上一轮对照已折叠；句内标点与接话间隔沿用，响度测量及固定增益另存。")
    elif manifest.get("revision", 1) >= 3:
        decision = manifest["feedback"]["decision"]
        print(f"已处理的最新意见：{decision['comments']}")
        print("优先听两份“本轮”音频：自然旁白、多角色接话。上一轮仅供可选回听。")
        print("本轮降低旁白情绪、恢复连贯标点，并按角色切换设置总间隔。")
    elif manifest.get("revision", 1) >= 2:
        decision = manifest["feedback"]["decision"]
        print(f"已处理的意见：{decision['comments']}；演绎强度：更克制。")
        print("优先听：上一轮演绎 → 轻演绎保持原分段 → 轻演绎连续旁白。")
        print("再比较角色衔接；平稳参考沿用旧音频，可跳过重复试听。")
    else:
        print("两条旁白各有原文平读、改写后平读、可控演绎；另一条为多角色演示。")
        print("平稳段落的后两组使用相同平静指令，作为避免过度渲染的对照。")
    print("音色暂用现有三份合成参考；稿件修订和来源记录保留。")
    print("已检查音频结构、非静音、请求及处理记录；听感、漏读与音色一致性待试听。")
    training = latest("latest_voxcpm_style_run.json")
    print(f"方案A第二版文本训练：{training if training else '执行中或尚未完成'}")
    print("本页面只读，不启动VoxCPM2、不执行训练。")
    return root


def show_audio():
    """在 Notebook 内嵌真实音频，用户无需寻找文件或重新合成。"""
    from html import escape

    from IPython.display import HTML, Audio, Markdown, display

    root = latest("latest_voxcpm_audition.json")
    if root is None:
        raise FileNotFoundError("尚无完整试听")
    manifest = json.loads((root / "manifest.json").read_text())
    bundle = json.loads(
        Path(manifest.get("bundle_file", PROJECT_ROOT / "eval/voxcpm_audition_v1.json")).read_text()
    )
    entries = manifest["entries"]
    # 先听同音色对照，再听多角色示范，避免切换音色影响文字对照判断。
    ordered_ids = bundle["comparison_ids"] + bundle["role_demo_ids"]
    if "audition_order" in bundle:
        ordered_ids = bundle["audition_order"]
    elif manifest.get("revision", 1) >= 2:
        ordered_ids = [
            identifier
            for identifier in ["sus_test_06", "sus_test_04", "sus_test_08"]
            if any(e["id"] == identifier for e in entries)
        ]
    for number, identifier in enumerate(ordered_ids, 1):
        case = next(c for c in bundle["cases"] if c["id"] == identifier)
        heading = (
            f"试听 {number}：{case['title']}" if manifest.get("revision", 1) >= 5 else case["title"]
        )
        display(Markdown(f"**{heading}**\n\n原文：{case['source_text']}"))
        if manifest.get("revision", 1) >= 5:
            display(
                Markdown(
                    "关注："
                    + case["listening_focus"]
                    + "\n\n只需记录有问题的编号与现象；旧版可按需展开。"
                )
            )
            display(
                HTML(
                    "<details><summary>查看角色与语气稿</summary><pre>"
                    + escape(case["script"])
                    + "</pre></details>"
                )
            )
        elif manifest.get("revision", 1) >= 4:
            display(
                Markdown(
                    "本轮语气稿（正文及标点沿用）：\n\n```text\n"
                    + case["script"]
                    + "\n```\n\n优先听本轮：旁白起伏是否合适、对白是否还夸张、角色音量切换是否突兀。旧版对照已折叠，可按需展开。"
                )
            )
        elif manifest.get("revision", 1) >= 3:
            display(
                Markdown(
                    "本轮人工修订稿（旧音频使用上一轮稿件）：\n\n```text\n"
                    + case["script"]
                    + "\n```\n\n本轮同时调整语气、标点和接话间隔；优先听“本轮”，旧版可选回听。"
                )
            )
        else:
            display(Markdown("演绎稿：\n\n```text\n" + case["script"] + "\n```"))
        case_entries = [e for e in entries if e["id"] == identifier]
        if manifest.get("revision", 1) >= 3:
            case_entries.sort(
                key=lambda e: e.get("optional_comparison", bool(e.get("reused_from")))
            )
        for entry in case_entries:
            label = f"{entry['label']} · {entry['audio']['duration_seconds']}秒"
            player = Audio(filename=str(root / entry["audio_file"]), embed=True)
            if manifest.get("revision", 1) >= 4 and entry.get(
                "optional_comparison", bool(entry.get("reused_from"))
            ):
                display(
                    HTML(
                        "<details><summary>"
                        + escape(label)
                        + "</summary>"
                        + player._repr_html_()
                        + "</details>"
                    )
                )
            else:
                display(Markdown(label))
                display(player)


def show_training():
    """显示本轮反馈处理评审；没有新反馈报告时查看文本训练。"""
    from IPython.display import Markdown, display

    audition = latest("latest_voxcpm_audition.json")
    if audition is not None and (audition / "feedback_review.md").exists():
        metadata = json.loads((audition / "manifest.json").read_text())
        if metadata.get("revision", 1) >= 6:
            current = [
                e
                for e in metadata["entries"]
                if not e.get("optional_comparison", bool(e.get("reused_from")))
            ]
            rows = "\n".join(
                f"| {i} | {e['title']} | {e['audio']['duration_seconds']} |"
                for i, e in enumerate(current, 1)
            )
            display(
                Markdown(
                    "**本轮已完成**：1、2按语气分段加强；3、4使用克制的日常对白，平静段落保持平和。已认可的语速处理和接话目标保留。\n\n"
                    "已确认1、2原先只有整段轻微起伏指令；3、4的平静标签没有变成紧张，但曾有迟疑片段被提高约16dB。现已限制正增益，并保留轻声与平静的音量差。\n\n"
                    "3、4附有折叠的“只改响度”对照：它复用旧合成，仅调整音量，可按需帮助判断用力感的来源。参考音色和韵律的影响尚需试听，尚未唯一归因。\n\n"
                    "| 编号 | 内容 | 时长/秒 |\n| --- | --- | --- |\n"
                    + rows
                    + "\n\n**只需写有问题的编号和现象**，例如“1情绪合适；3平静仍用力；4已放松；其余可沿用”。编号5是上一轮编号6正文的纯平静对照。新表单沿用本轮实际强度，三个字段都能修改；保存才提交下一轮意见。"
                )
            )
            return
        if metadata.get("revision", 1) >= 5:
            current = [e for e in metadata["entries"] if not e.get("reused_from")]
            rows = "\n".join(
                f"| {i} | {e['title']} | {e['audio']['duration_seconds']} |"
                for i, e in enumerate(current, 1)
            )
            display(
                Markdown(
                    "**本轮已完成**：语速轻微放缓，角色音量与接话间隔沿用；5份新例子已生成。\n\n| 编号 | 内容 | 时长/秒 |\n| --- | --- | --- |\n"
                    + rows
                    + "\n\n先判断速度，再判断情绪转折、接话与音量。**只需写有问题的编号和现象**，其余可写“可沿用”。\n\n新例子由agent编写审阅，用于声音控制试听；没有重训文本LoRA，也不作为新模型预测成绩。"
                )
            )
            return
        display(Markdown((audition / "feedback_review.md").read_text(encoding="utf-8")))
        return
    root = latest("latest_voxcpm_style_run.json")
    if root is None:
        print("第二版训练尚未完成。重复运行此格只刷新显示，不启动训练。")
        return
    for name in ("agent_review.md", "summary.md"):
        path = root / name
        if path.exists():
            display(Markdown(path.read_text(encoding="utf-8")))


def decision_widget():
    """只有点击保存才写监督意见；不会据此自动重训或批准批量使用。"""
    import ipywidgets as widgets
    from IPython.display import display

    root = latest("latest_voxcpm_audition.json")
    if root is None:
        raise FileNotFoundError("尚无完整试听")
    from voxcpm_adapter import load_delivery

    saved = root / "user_decision.json"
    existing = json.loads(saved.read_text()) if saved.exists() else {}
    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    strength = widgets.Dropdown(
        options=[("更克制", "mild"), ("当前适度", "moderate"), ("更明显", "strong")],
        value=existing.get(
            "strength", manifest.get("resolved_strength", load_delivery()["default_strength"])
        ),
        description="演绎强度：",
    )
    verdict = widgets.Dropdown(
        options=[("继续调整", "revise"), ("可沿用方向", "continue")],
        value=existing.get("verdict", "revise"),
        description="监督意见：",
    )
    comments = widgets.Textarea(
        description="备注：",
        placeholder="只需写有问题的编号和现象，例如：1情绪太弱；3平静仍用力；4已放松；其余可沿用。",
        layout=widgets.Layout(width="95%"),
        value=existing.get("comments", ""),
    )
    button = widgets.Button(description="保存试听意见", button_style="primary")
    status = widgets.Output()

    def save_decision(_):
        from datetime import UTC, datetime

        decision = {
            "created_at": datetime.now(UTC).isoformat(),
            "strength": strength.value,
            "verdict": verdict.value,
            "comments": comments.value,
            "scope": "用户对本轮试听的意见；不等同于批量内容或模型正式上线批准",
        }
        path = root / "user_decision.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        with status:
            status.clear_output()
            print(f"已保存：{path}。你也可以直接在对话中告诉我意见。")

    button.on_click(save_decision)
    display(widgets.VBox([strength, verdict, comments, button, status]))


if __name__ == "__main__":
    show_status()
