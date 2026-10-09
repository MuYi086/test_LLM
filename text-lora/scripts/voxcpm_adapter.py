"""把已审阅的演绎稿映射到本地 VoxCPM2 可控克隆请求。"""

import hashlib
import json
import re
from pathlib import Path

LINE = re.compile(r"^\[([^\[\]\n]+)\]\[([^\[\]\n]+)\](\S.*)$")
ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value):
    """记录稿件、配置与请求的内容指纹，防止误复用旧样音。"""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def load_delivery(path=ROOT / "configs/voxcpm_delivery.json"):
    """读取语气词典；控制词仍需用户试听，不能视作声音效果保证。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["schema_version"] != 1 or data["max_segment_chars"] <= 0:
        raise ValueError("演绎配置无效")
    for tone, spec in data["tones"].items():
        if not tone or not spec["instruction"].strip() or not 0 <= spec["pause_ms"] <= 2000:
            raise ValueError("演绎语气配置无效")
    return data


def parse_script(script, roles, tones):
    """拒绝未知标签和模型思考内容，避免把控制标记读入声音。"""
    rows = []
    for raw in script.splitlines():
        if not raw.strip():
            continue
        match = LINE.fullmatch(raw.strip())
        if not match:
            raise ValueError(f"无效演绎行：{raw}")
        speaker, tone, text = match.groups()
        if speaker not in roles or tone not in tones:
            raise ValueError(f"未知人物或语气：{speaker}/{tone}")
        if any(marker in text for marker in ("[", "]", "<think", "</think", "```")):
            raise ValueError("正文夹带控制标记或思考块")
        rows.append({"speaker": speaker, "tone": tone, "text": text})
    if not rows:
        raise ValueError("演绎稿为空")
    return rows


def verify_review(case, rows):
    """核对稿件审阅指纹、声明的事实及对白；不声称证明全部语义。"""
    reviewed = {key: case[key] for key in ("source_text", "script", "allowed_roles")}
    if case.get("review_status") != "agent_checked" or case.get("review_sha256") != fingerprint(
        reviewed
    ):
        raise ValueError("需要与当前原文及演绎稿一致的 agent 审阅记录")
    for fact in case["fact_checks"]:
        if not fact["source"] or fact["source"] not in case["source_text"]:
            raise ValueError("事实检查没有原文来源")
        text = "\n".join(r["text"] for r in rows if fact.get("role") in (None, r["speaker"]))
        if not fact["target"] or fact["target"] not in text:
            raise ValueError(f"声明的事实缺失或角色错误：{fact['target']}")
    for speech in case["speech_checks"]:
        actual = sum(
            r["speaker"] == speech["speaker"] and r["text"] == speech["text"] for r in rows
        )
        total = sum(r["text"].count(speech["text"]) for r in rows)
        if (
            actual != speech["count"]
            or total != speech["count"]
            or case["source_text"].count(speech["text"]) != speech["count"]
        ):
            raise ValueError(f"直接对白归属或次数错误：{speech['text']}")


def compile_segments(rows, voices, delivery, strength="moderate", controlled=True):
    """合并相邻同人物同语气的稿件，保留边界停顿并生成合法请求。"""
    if strength not in delivery["strength_suffix"]:
        raise ValueError("未知演绎强度")
    merged = []
    limit = delivery["max_segment_chars"]
    for row in rows:
        if row["speaker"] not in voices or not voices[row["speaker"]]:
            raise ValueError(f"人物没有参考音色：{row['speaker']}")
        if len(row["text"]) > limit:
            raise ValueError("单行超过合成长度限制，需要先按句拆分和审阅")
        key = (row["speaker"], row["tone"] if controlled else "平静")
        if (
            merged
            and merged[-1]["key"] == key
            and len(merged[-1]["text"]) + len(row["text"]) <= limit
        ):
            merged[-1]["text"] += row["text"]
        else:
            merged.append({"key": key, "text": row["text"]})
    result = []
    for group in merged:
        speaker, tone = group["key"]
        spec = delivery["tones"][tone]
        instruction = spec["instruction"] if controlled else delivery["neutral_instruction"]
        if controlled and tone not in {"平静", "冷静"}:
            instruction += delivery["strength_suffix"][strength]
        result.append(
            {
                "speaker": speaker,
                "tone": tone,
                "pause_after_ms": spec["pause_ms"],
                "request": {
                    "text": group["text"],
                    "audio_path": voices[speaker],
                    "clone_mode": "controllable",
                    "control_instruction": instruction,
                    "seed": delivery["seed"],
                    "max_chars_per_chunk": 0,
                },
            }
        )
    return result
