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
    print("两条旁白各有原文平读、改写后平读、可控演绎；另一条为多角色演示。")
    print("平稳段落的后两组使用相同平静指令，作为避免过度渲染的对照。")
    print("音色暂用现有三份合成参考；平稳段落已由 agent 修正语气，来源记录保留。")
    print("目前只检查了WAV结构、非静音和请求协议；听感、漏读与音色一致性待试听。")
    training = latest("latest_voxcpm_style_run.json")
    print(f"方案A第二版文本训练：{training if training else '执行中或尚未完成'}")
    print("本页面只读，不启动VoxCPM2、不执行训练。")
    return root


def show_audio():
    """在 Notebook 内嵌真实音频，用户无需寻找文件或重新合成。"""
    from IPython.display import Audio, Markdown, display

    root = latest("latest_voxcpm_audition.json")
    if root is None:
        raise FileNotFoundError("尚无完整试听")
    manifest = json.loads((root / "manifest.json").read_text())
    bundle = json.loads((PROJECT_ROOT / "eval/voxcpm_audition_v1.json").read_text())
    entries = manifest["entries"]
    # 先听同音色对照，再听多角色示范，避免切换音色影响文字对照判断。
    ordered_ids = bundle["comparison_ids"] + bundle["role_demo_ids"]
    for identifier in ordered_ids:
        case = next(c for c in bundle["cases"] if c["id"] == identifier)
        display(Markdown(f"**{case['title']}**\n\n原文：{case['source_text']}"))
        display(Markdown("演绎稿：\n\n```text\n" + case["script"] + "\n```"))
        for entry in [e for e in entries if e["id"] == identifier]:
            display(Markdown(f"{entry['label']} · {entry['audio']['duration_seconds']}秒"))
            display(Audio(filename=str(root / entry["audio_file"]), embed=True))


def show_training():
    """显示第二版三组文本对照，训练指标与声音试听分开解释。"""
    from IPython.display import Markdown, display

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
    strength = widgets.Dropdown(
        options=[("更克制", "mild"), ("当前适度", "moderate"), ("更明显", "strong")],
        value="moderate",
        description="演绎强度：",
    )
    verdict = widgets.Dropdown(
        options=[("继续调整", "revise"), ("可沿用方向", "continue")],
        value="revise",
        description="监督意见：",
    )
    comments = widgets.Textarea(
        description="备注：",
        placeholder="例如：紧张感合适，但旁白偏慢；角色音色需要更换。",
        layout=widgets.Layout(width="95%"),
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
