"""从已完成实验中整理两条旁白对照与一条角色演绎试听。"""

import json
import re
from pathlib import Path

from check_environment import PROJECT_ROOT
from voxcpm_adapter import fingerprint, load_delivery, parse_script, verify_review

AUDITION = PROJECT_ROOT / "eval/voxcpm_audition_v1.json"


def prepare():
    """保留原始模型记录，另存已审阅或经 agent 修订的试听稿。"""
    root = Path(
        json.loads((PROJECT_ROOT / "outputs/latest_suspense_run.json").read_text())["run_dir"]
    )
    records = {
        r["id"]: r for r in map(json.loads, (root / "style/results.jsonl").read_text().splitlines())
    }
    benchmark = json.loads((PROJECT_ROOT / "eval/suspense_benchmark_v1.json").read_text())
    cases = []
    for reference in benchmark["cases"]:
        if reference["id"] not in {"sus_test_04", "sus_test_06", "sus_test_08"}:
            continue
        case = {
            key: reference[key]
            for key in (
                "id",
                "title",
                "source_text",
                "allowed_roles",
                "fact_checks",
                "speech_checks",
            )
        }
        output = records[case["id"]]["output"]
        script = re.sub(r"^<think>\s*</think>\s*", "", output).strip()
        provenance = "existing_lora_output_agent_reviewed"
        if case["id"] == "sus_test_08":
            script = "\n".join(
                [
                    "[旁白][平静]天还没黑。",
                    "[旁白][平静]池岳在屋外收好晾着的毛巾，把竹篮放到门边。",
                    "[旁白][平静]他逐一关上窗，最后将门扣好。",
                    "[旁白][平静]屋里的钟走得很准。桌上的茶还温着。",
                    "[旁白][平静]他坐下，开始整理当天的账目。",
                ]
            )
            provenance = "existing_lora_output_tones_corrected_by_agent"
        # 独立检查使用当前已审阅稿中的准确片段；保留来源检查，不改旧 benchmark。
        if case["id"] == "sus_test_06":
            for fact in case["fact_checks"]:
                if fact["source"] == "比上一帧靠左":
                    fact["target"] = "比上一帧靠左"
                if fact["source"] == "没有把这个想法说出去":
                    fact["target"] = "没有把这个想法说出去"
        if case["id"] == "sus_test_08":
            for fact in case["fact_checks"]:
                if fact["source"] == "逐一关上窗":
                    fact["target"] = "逐一关上窗"
        case.update(
            script=script,
            provenance=provenance,
            model_run=str(root),
            review_status="agent_checked",
            user_audio_decision="pending",
        )
        case["review_sha256"] = fingerprint(
            {key: case[key] for key in ("source_text", "script", "allowed_roles")}
        )
        rows = parse_script(script, case["allowed_roles"], load_delivery()["tones"])
        verify_review(case, rows)
        cases.append(case)
    bundle = {
        "schema_version": 1,
        "purpose": "方案A小样试听；两条单旁白三组对照与一条多角色示范，不用于训练或音质排名",
        "comparison_ids": ["sus_test_06", "sus_test_08"],
        "role_demo_ids": ["sus_test_04"],
        "provisional_voices": {
            "旁白": "qwen_voicedesign_20261006_145406_r0dpazut.wav",
            "洛洵": "qwen_voicedesign_20261006_145447_t9gx5tew.wav",
            "袁星": "qwen_voicedesign_20261006_145527_2a6z4orn.wav",
        },
        "voice_note": "暂用用户现有的三份合成参考音色；不推断音色性别，不代表最终角色选角",
        "cases": cases,
    }
    encoded = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    if AUDITION.exists() and AUDITION.read_text(encoding="utf-8") != encoded:
        raise ValueError("试听输入已固定，修改时请另建版本")
    AUDITION.write_text(encoded, encoding="utf-8")
    print(f"[PASS] 3条稿件已核对事实、对白和语气边界：{AUDITION}")
    return bundle


if __name__ == "__main__":
    prepare()
